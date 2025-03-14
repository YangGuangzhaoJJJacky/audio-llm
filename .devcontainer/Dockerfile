FROM mcr.microsoft.com/devcontainers/base:ubuntu

# 安装基本系统依赖
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && uv tool install ruff

# 设置工作目录
WORKDIR /workspace

# 设置git安全目录，避免权限问题
RUN git config --global --add safe.directory /workspace

# 设置环境变量
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
CMD [ "uv", "--version" ]