#!/usr/bin/env python3
"""
Organize downloaded malware and benign samples for PackHero
Creates proper directory structure and metadata
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from datetime import datetime

class SampleOrganizer:
    """Organize malware and benign samples"""
    
    def __init__(self, base_dir="packhero-dataset"):
        self.base_dir = Path(base_dir)
        self.stats = {
            'malware': 0,
            'benign': 0,
            'duplicates': 0,
            'errors': 0
        }
        
    def setup_directories(self):
        """Create directory structure"""
        print("Creating directory structure...")
        
        dirs = [
            'original/malware',
            'original/benign',
            'packed/upx/malware',
            'packed/upx/benign',
            'packed/upx_best/malware',
            'packed/upx_best/benign',
            'packed/upx_ultra/malware',
            'packed/upx_ultra/benign',
            'labels',
            'metadata'
        ]
        
        for dir_path in dirs:
            (self.base_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✓ Directory structure created")
    
    def get_file_hash(self, filepath):
        """Calculate SHA256 hash"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def is_pe_file(self, filepath):
        """Check if file is a valid PE executable"""
        try:
            with open(filepath, 'rb') as f:
                header = f.read(2)
                return header == b'MZ'  # DOS/PE header
        except:
            return False
    
    def organize_samples(self, source_dir, sample_type='malware'):
        """
        Organize samples from source directory
        
        Args:
            source_dir: Directory containing .exe files
            sample_type: 'malware' or 'benign'
        """
        source_path = Path(source_dir)
        dest_path = self.base_dir / 'original' / sample_type
        
        if not source_path.exists():
            print(f"✗ Source directory not found: {source_path}")
            return
        
        print(f"\nOrganizing {sample_type} samples from: {source_path}")
        
        # Track hashes to detect duplicates
        seen_hashes = set()
        
        # Get all .exe files
        exe_files = list(source_path.glob('*.exe'))
        
        if not exe_files:
            print(f"  No .exe files found in {source_path}")
            return
        
        for idx, file_path in enumerate(exe_files, 1):
            try:
                # Validate PE file
                if not self.is_pe_file(file_path):
                    print(f"  [{idx}/{len(exe_files)}] ✗ Not a valid PE file: {file_path.name}")
                    self.stats['errors'] += 1
                    continue
                
                # Calculate hash
                file_hash = self.get_file_hash(file_path)
                
                # Check for duplicates
                if file_hash in seen_hashes:
                    print(f"  [{idx}/{len(exe_files)}] ⊙ Duplicate: {file_path.name}")
                    self.stats['duplicates'] += 1
                    continue
                
                seen_hashes.add(file_hash)
                
                # Create new filename with hash prefix
                new_name = f"{sample_type}_{file_hash[:16]}_{file_path.name}"
                dest_file = dest_path / new_name
                
                # Copy file
                shutil.copy2(file_path, dest_file)
                
                print(f"  [{idx}/{len(exe_files)}] ✓ {new_name}")
                self.stats[sample_type] += 1
                
            except Exception as e:
                print(f"  [{idx}/{len(exe_files)}] ✗ Error processing {file_path.name}: {e}")
                self.stats['errors'] += 1
    
    def generate_metadata(self):
        """Generate metadata about the dataset"""
        metadata = {
            'created': datetime.now().isoformat(),
            'structure': 'PackHero dataset',
            'statistics': self.stats.copy()
        }
        
        # Count files in each directory
        for sample_type in ['malware', 'benign']:
            orig_path = self.base_dir / 'original' / sample_type
            if orig_path.exists():
                count = len(list(orig_path.glob('*.exe')))
                metadata[f'{sample_type}_count'] = count
        
        # Save metadata
        metadata_file = self.base_dir / 'metadata' / 'dataset_info.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved to: {metadata_file}")
        return metadata
    
    def print_summary(self):
        """Print organization summary"""
        print("\n" + "="*60)
        print("Organization Summary")
        print("="*60)
        print(f"Malware samples organized: {self.stats['malware']}")
        print(f"Benign samples organized: {self.stats['benign']}")
        print(f"Duplicates skipped: {self.stats['duplicates']}")
        print(f"Errors encountered: {self.stats['errors']}")
        print(f"Total samples: {self.stats['malware'] + self.stats['benign']}")
        print("="*60)
        
        # Check if we have enough samples
        total = self.stats['malware'] + self.stats['benign']
        if total < 10:
            print("\n⚠️  Warning: You have fewer than 10 samples.")
            print("   PackHero needs at least 10 samples per packer.")
            print("   Consider downloading more samples.")
        elif total < 50:
            print("\n✓ Good start! You have enough for basic testing.")
            print("  For better results, aim for 50-100 samples total.")
        else:
            print("\n✓ Excellent! You have enough samples for proper training.")
        
        print(f"\n📁 Dataset location: {self.base_dir.absolute()}")
        print("\nNext steps:")
        print("  1. Pack samples with UPX: bash pack_with_upx.sh")
        print("  2. Generate labels: python3 generate_labels.py")
        print("  3. Configure PackHero with your samples")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Organize malware and benign samples for PackHero',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Organize malware samples
  python3 organize_samples.py --malware ./malware_samples --output ./packhero-dataset
  
  # Organize benign samples
  python3 organize_samples.py --benign ./benign_samples --output ./packhero-dataset
  
  # Organize both
  python3 organize_samples.py --malware ./malware --benign ./benign
        """
    )
    
    parser.add_argument(
        '--malware',
        type=str,
        help='Directory containing malware .exe files'
    )
    
    parser.add_argument(
        '--benign',
        type=str,
        help='Directory containing benign .exe files'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='packhero-dataset',
        help='Output base directory (default: packhero-dataset)'
    )
    
    args = parser.parse_args()
    
    if not args.malware and not args.benign:
        parser.error("Provide at least --malware or --benign directory")
    
    # Create organizer
    organizer = SampleOrganizer(base_dir=args.output)
    
    # Setup directories
    organizer.setup_directories()
    
    # Organize samples
    if args.malware:
        organizer.organize_samples(args.malware, sample_type='malware')
    
    if args.benign:
        organizer.organize_samples(args.benign, sample_type='benign')
    
    # Generate metadata
    organizer.generate_metadata()
    
    # Print summary
    organizer.print_summary()

if __name__ == "__main__":
    main()
