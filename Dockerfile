# 1. 直接基于本地已有的 cmu-lab 镜像
FROM cmu-lab:latest

# 2. 换成国内清华大学 apt 镜像源
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's@security.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

# 3. 安装 Proj4 编译所必需的头文件与工具库
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    libcunit1-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 4. 安装测试所需的 Python 包（通过清华源安装适配 Python 3.6 的版本）
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    numpy==1.19.5 \
    pytest

# 5. 做软链接，兼容 Makefile 里写死的伯克利 CUnit 路径，无需修改原始 Makefile
RUN mkdir -p /home/ff/cs61c/cunit/install && \
    ln -s /usr/include /home/ff/cs61c/cunit/install/include && \
    ln -s /usr/lib/x86_64-linux-gnu /home/ff/cs61c/cunit/install/lib

WORKDIR /workspace
