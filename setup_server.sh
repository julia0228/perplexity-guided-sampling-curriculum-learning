#!/usr/bin/env bash
# 在服务器上执行此脚本: bash setup_server.sh
# 前提: 已 git clone 本 repo，CUDA 驱动正常

set -euo pipefail

# ====== 配置 ======
REPO_DIR=$(cd "$(dirname "$0")" && pwd)
CONDA_ENV_NAME="sec"
PYTHON_VERSION="3.10"
CUDA_VERSION="cu121"  # 根据你的 CUDA 版本修改: cu118, cu121, cu124

# ====== 1. 创建 conda 环境 ======
echo ">>> 创建 conda 环境: ${CONDA_ENV_NAME}"
conda create -n "${CONDA_ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV_NAME}"

# ====== 2. 安装 PyTorch ======
echo ">>> 安装 PyTorch (${CUDA_VERSION})"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"

# ====== 3. 安装 flash-attn（先装，避免后面编译卡住）======
echo ">>> 安装 flash-attn"
pip install flash-attn --no-build-isolation

# ====== 4. 安装 xformers ======
echo ">>> 安装 xformers"
pip install xformers

# ====== 5. 安装项目依赖 ======
echo ">>> 安装 verl 项目"
cd "${REPO_DIR}/verl"
pip install -e .

# ====== 6. 登录 wandb ======
echo ">>> 请手动执行 wandb login 并输入你的 API key"

echo ""
echo "===== 安装完成 ====="
echo "激活环境: conda activate ${CONDA_ENV_NAME}"
echo "运行训练: bash examples/countdown.sh countdown_test"
