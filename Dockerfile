FROM ubuntu:22.04

# 避免 apt 安装过程中出现交互式时区等弹窗
ENV DEBIAN_FRONTEND=noninteractive
# 绕过 Ubuntu 22.04 中 PEP 668 环境限制，允许系统级 pip 安装
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# 替换 APT 源为清华镜像以加速安装
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 安装系统级编译依赖、Python 开发环境和 CUnit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    libcunit1-dev \
    && rm -rf /var/lib/apt/lists/*

# 配置清华 PyPI 镜像并安装 Python 依赖
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --no-cache-dir \
    numpy \
    pytest \
    setuptools \
    wheel

# 设定工作区目录
WORKDIR /workspace
