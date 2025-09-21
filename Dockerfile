# Use NVIDIA CUDA base image for GPU support (Ubuntu 22.04)
FROM nvidia/cuda:12.8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Update package lists and install essential packages
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    wget \
    vim \
    git \
    openssh-server \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit and GPU monitoring tools
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    nvidia-utils-550 \
    && rm -rf /var/lib/apt/lists/*

# Set locale to UTF-8
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8

# Ensure NVIDIA libraries are in LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# Create a non-root user with sudo privileges
RUN groupadd -g 1000 vscode && \
    useradd -u 1000 -g vscode -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set ownership for home directory
RUN chown -R vscode:vscode /home/vscode

# Switch to vscode user
USER vscode
WORKDIR /home/vscode

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/home/vscode/.local/bin:$PATH"

# Install GitHub CLI
RUN (type -p wget >/dev/null || (sudo apt-get update && sudo apt-get install wget -y)) \
    && mkdir -p -m 755 /tmp/apt-keyrings \
    && out=$(mktemp) && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && cat "$out" | gpg --dearmor | sudo tee /tmp/apt-keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && sudo chmod go+r /tmp/apt-keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/tmp/apt-keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && sudo apt-get update \
    && sudo apt-get install gh -y \
    && rm -rf /tmp/apt-keyrings

# Set up SSH for vscode user
RUN sudo mkdir -p /run/sshd && \
    sudo mkdir -p /home/vscode/.ssh && \
    sudo chmod 700 /home/vscode/.ssh && \
    sudo chown vscode:vscode /home/vscode/.ssh

# Configure SSH to allow password authentication and key-based authentication
RUN sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin no/' /etc/ssh/sshd_config && \
    sudo sed -i 's/#UsePAM yes/UsePAM yes/' /etc/ssh/sshd_config

# Generate SSH host keys (will be persisted via volume mount)
RUN sudo ssh-keygen -A

# Create authorized_keys file
RUN touch /home/vscode/.ssh/authorized_keys && \
    sudo chmod 600 /home/vscode/.ssh/authorized_keys && \
    sudo chown vscode:vscode /home/vscode/.ssh/authorized_keys

# Set up project directory with proper permissions
RUN sudo mkdir -p /workspaces && \
    sudo chown -R vscode:vscode /workspaces && \
    sudo chmod -R 755 /workspaces

# Expose SSH port
EXPOSE 22

# Set working directory to project
WORKDIR /workspaces

# Start SSH daemon when container starts
CMD ["/usr/sbin/sshd", "-D"]