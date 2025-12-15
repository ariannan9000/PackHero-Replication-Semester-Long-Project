#!/usr/bin/env python3
"""
Graph extraction script for PackHero.

Extracts call graphs from PE executables using radare2 and converts them
to PyTorch Geometric format for GNN training.

Usage:
    python -m gnn.extract_graphs --input packhero-dataset/packed --output packhero-dataset/graphs

Requirements:
    - radare2 (https://rada.re/n/radare2.html)
    - r2pipe (pip install r2pipe)

Node Features (5 total):
    1. size: Function size in bytes
    2. blocks: Number of basic blocks
    3. instructions: Number of instructions
    4. outgoing_calls: Number of calls to other functions
    5. is_entry: Whether this is the entry point (0 or 1)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import torch
from torch_geometric.data import Data

try:
    import r2pipe
    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False
    print("Warning: r2pipe not installed. Install with: pip install r2pipe")


def extract_call_graph_r2(pe_path: str) -> Optional[Dict]:
    """
    Extract call graph from a PE file using radare2.

    Args:
        pe_path: Path to the PE executable

    Returns:
        Dictionary containing nodes and edges, or None on failure
    """
    if not R2_AVAILABLE:
        raise ImportError("r2pipe is required for graph extraction")

    try:
        # Open file with radare2
        r2 = r2pipe.open(pe_path, flags=["-2"])  # -2 for no stderr

        # Analyze the binary
        r2.cmd("aaa")  # Full analysis

        # Get function list
        functions_json = r2.cmd("aflj")
        if not functions_json:
            r2.quit()
            return None

        functions = json.loads(functions_json)

        if not functions:
            r2.quit()
            return None

        # Create node mapping
        func_to_idx = {}
        nodes = []

        for idx, func in enumerate(functions):
            addr = func.get("offset", 0)
            func_to_idx[addr] = idx

            # Extract node features
            size = func.get("size", 0)
            nblocks = func.get("nbb", func.get("nbbs", 1))  # Number of basic blocks
            ninstrs = func.get("ninstrs", func.get("ninstr", 0))  # Number of instructions
            is_entry = 1 if func.get("name", "").startswith("entry") else 0

            nodes.append({
                "idx": idx,
                "addr": addr,
                "name": func.get("name", f"fcn_{addr:x}"),
                "size": size,
                "blocks": nblocks,
                "instructions": ninstrs,
                "is_entry": is_entry,
            })

        # Get call references (edges)
        edges = []
        for func in functions:
            addr = func.get("offset", 0)
            src_idx = func_to_idx.get(addr)

            if src_idx is None:
                continue

            # Get cross-references from this function
            xrefs_json = r2.cmd(f"axtj @ {addr}")
            if xrefs_json:
                try:
                    xrefs = json.loads(xrefs_json)
                    for xref in xrefs:
                        if xref.get("type") == "CALL":
                            target_addr = xref.get("from", 0)
                            dst_idx = func_to_idx.get(target_addr)
                            if dst_idx is not None and src_idx != dst_idx:
                                edges.append((src_idx, dst_idx))
                except json.JSONDecodeError:
                    pass

            # Also get calls from this function
            calls_json = r2.cmd(f"axfj @ {addr}")
            if calls_json:
                try:
                    calls = json.loads(calls_json)
                    for call in calls:
                        if call.get("type") == "CALL":
                            target_addr = call.get("to", call.get("addr", 0))
                            dst_idx = func_to_idx.get(target_addr)
                            if dst_idx is not None and src_idx != dst_idx:
                                edges.append((src_idx, dst_idx))
                except json.JSONDecodeError:
                    pass

        r2.quit()

        # Remove duplicate edges
        edges = list(set(edges))

        # Add outgoing_calls to nodes
        outgoing_counts = {}
        for src, _ in edges:
            outgoing_counts[src] = outgoing_counts.get(src, 0) + 1

        for node in nodes:
            node["outgoing_calls"] = outgoing_counts.get(node["idx"], 0)

        return {
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        }

    except Exception as e:
        print(f"Error extracting graph: {e}")
        return None


def graph_to_pyg(graph_data: Dict) -> Data:
    """
    Convert extracted graph to PyTorch Geometric Data object.

    Args:
        graph_data: Dictionary with nodes and edges

    Returns:
        PyTorch Geometric Data object
    """
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # Create node feature matrix
    # Features: [size, blocks, instructions, outgoing_calls, is_entry]
    features = []
    for node in nodes:
        features.append([
            float(node["size"]),
            float(node["blocks"]),
            float(node["instructions"]),
            float(node["outgoing_calls"]),
            float(node["is_entry"]),
        ])

    x = torch.tensor(features, dtype=torch.float)

    # Create edge index
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


def process_pe_file(
    pe_path: Path,
    output_dir: Path,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Process a single PE file: extract graph and save as .pt file.

    Args:
        pe_path: Path to PE file
        output_dir: Directory to save .pt file
        verbose: Whether to print progress

    Returns:
        Path to saved .pt file, or None on failure
    """
    if verbose:
        print(f"Processing: {pe_path.name}")

    # Extract call graph
    graph_data = extract_call_graph_r2(str(pe_path))

    if graph_data is None or graph_data["num_nodes"] == 0:
        if verbose:
            print(f"  Skipped: No functions found")
        return None

    # Convert to PyTorch Geometric
    data = graph_to_pyg(graph_data)

    # Save
    output_name = pe_path.stem + ".pt"
    output_path = output_dir / output_name
    torch.save(data, output_path)

    if verbose:
        print(f"  Saved: {output_path.name} ({data.num_nodes} nodes, {data.num_edges} edges)")

    return output_path


def process_directory(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Process all PE files in a directory.

    Args:
        input_dir: Input directory containing PE files
        output_dir: Output directory for .pt files
        recursive: Whether to search recursively
        verbose: Whether to print progress

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .exe files
    if recursive:
        pe_files = list(input_dir.rglob("*.exe"))
    else:
        pe_files = list(input_dir.glob("*.exe"))

    stats = {
        "total": len(pe_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
    }

    print(f"Found {len(pe_files)} PE files")

    for i, pe_path in enumerate(pe_files, 1):
        print(f"\n[{i}/{len(pe_files)}] ", end="")

        try:
            result = process_pe_file(pe_path, output_dir, verbose)
            if result:
                stats["processed"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            print(f"  Error: {e}")
            stats["errors"] += 1

    return stats


def create_synthetic_graphs(
    csv_path: Path,
    output_dir: Path,
    num_nodes_range: Tuple[int, int] = (5, 50),
    seed: int = 42,
) -> int:
    """
    Create synthetic graph files for testing when radare2 is not available.

    This generates random graphs matching the filenames in the CSV.
    Useful for testing the GNN pipeline without actual graph extraction.

    Args:
        csv_path: Path to the dataset CSV file
        output_dir: Output directory for .pt files
        num_nodes_range: Range of number of nodes (min, max)
        seed: Random seed

    Returns:
        Number of graphs created
    """
    import pandas as pd
    import random

    random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    count = 0

    for _, row in df.iterrows():
        filename = row["filename"]
        graph_name = Path(filename).stem + ".pt"
        output_path = output_dir / graph_name

        # Random number of nodes
        num_nodes = random.randint(*num_nodes_range)

        # Random features
        x = torch.randn(num_nodes, 5)
        # Make features more realistic
        x[:, 0] = torch.abs(x[:, 0]) * 1000  # size
        x[:, 1] = torch.abs(x[:, 1]) * 10    # blocks
        x[:, 2] = torch.abs(x[:, 2]) * 100   # instructions
        x[:, 3] = torch.abs(x[:, 3]) * 5     # outgoing_calls
        x[:, 4] = torch.zeros(num_nodes)     # is_entry
        x[0, 4] = 1.0  # First node is entry

        # Random edges (sparse call graph)
        num_edges = random.randint(num_nodes // 2, num_nodes * 2)
        edges = set()
        for _ in range(num_edges):
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst:
                edges.add((src, dst))

        if edges:
            edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        torch.save(data, output_path)
        count += 1

    print(f"Created {count} synthetic graph files in {output_dir}")
    return count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract call graphs from PE files for PackHero GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="packhero-dataset/packed",
        help="Input directory containing PE files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="packhero-dataset/graphs",
        help="Output directory for .pt graph files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search input directory recursively",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Create synthetic graphs for testing (when radare2 unavailable)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="packhero-dataset/labels/packhero_dataset.csv",
        help="CSV file for synthetic graph generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation",
    )

    args = parser.parse_args()

    if args.synthetic:
        # Create synthetic graphs for testing
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            sys.exit(1)

        create_synthetic_graphs(
            csv_path=csv_path,
            output_dir=Path(args.output),
            seed=args.seed,
        )
    else:
        # Extract real graphs using radare2
        if not R2_AVAILABLE:
            print("Error: r2pipe is required for graph extraction")
            print("Install with: pip install r2pipe")
            print("Also ensure radare2 is installed: https://rada.re/n/radare2.html")
            print("\nAlternatively, use --synthetic to create test graphs")
            sys.exit(1)

        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)

        output_dir = Path(args.output)

        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print("-" * 50)

        stats = process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            recursive=args.recursive,
        )

        print("\n" + "=" * 50)
        print("Graph Extraction Summary")
        print("=" * 50)
        print(f"Total PE files: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"Errors: {stats['errors']}")
        print(f"\nGraphs saved to: {output_dir}")


if __name__ == "__main__":
    main()
