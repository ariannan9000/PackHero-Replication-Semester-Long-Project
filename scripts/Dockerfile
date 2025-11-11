FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    wget \
    curl \
    git \
    file \
    unzip \
    upx-ucl \
    && rm -rf /var/lib/apt/limes/*

# Install Python packages
RUN pip3 install --no-cache-dir requests

# Create working directory
WORKDIR /workspace
RUN mkdir -p malware_samples benign_samples packhero-dataset

# Create a simple startup script
RUN echo '#!/bin/sh' > /start.sh && \
    echo 'echo "========================================"' >> /start.sh && \
    echo 'echo "  ISOLATED PACKHERO ENVIRONMENT"' >> /start.sh && \
    echo 'echo "========================================"' >> /start.sh && \
    echo 'echo "Status: Ready for malware analysis"' >> /start.sh && \
    echo 'echo ""' >> /start.sh && \
    echo 'echo "Platform: $(uname -m)"' >> /start.sh && \
    echo 'echo "Python: $(python3 --version 2>/dev/null)"' >> /start.sh && \
    echo 'echo "UPX: $(upx --version 2>/dev/null | head -1)"' >> /start.sh && \
    echo 'echo ""' >> /start.sh && \
    echo 'echo "Your Mac folder syncs with:"' >> /start.sh && \
    echo 'echo "  /workspace/data/"' >> /start.sh && \
    echo 'echo ""' >> /start.sh && \
    echo 'echo "Tools available:"' >> /start.sh && \
    echo 'echo "  python3, upx, wget, curl, git"' >> /start.sh && \
    echo 'echo "========================================"' >> /start.sh && \
    echo 'echo ""' >> /start.sh && \
    echo 'echo "Type your commands below:"' >> /start.sh && \
    echo 'exec /bin/sh' >> /start.sh && \
    chmod +x /start.sh

CMD ["/start.sh"]
