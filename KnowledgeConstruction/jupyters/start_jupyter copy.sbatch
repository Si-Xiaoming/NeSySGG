#!/bin/bash
#SBATCH --job-name=start_jupyter
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=d119918@studenti.polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=131072
#SBATCH --ntasks=32
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err



source activate promptllm
WORK_DIR="/home/shsi/codes/LLM/promptLLM"
cd "$WORK_DIR"
export PYTHONPATH=./
echo "Current directory: $(pwd)"

# 2. 获取计算节点信息
# 获取节点的主机名 (例如 node05)
NODE_NAME=$(hostname)
# 随机生成一个端口号 (10000-60000之间)，防止端口冲突
PORT=$(shuf -i 10000-60000 -n 1)

# 3. 打印连接指南 (这是最重要的一步，输出到日志文件)
echo "========================================================================"
echo "    JUPYTER NOTEBOOK IS RUNNING ON NODE: $NODE_NAME"
echo "========================================================================"
echo ""
echo "Step 1: On your LOCAL computer (laptop), run this command:"
echo ""
echo "    ssh -N -L ${PORT}:${NODE_NAME}:${PORT} hpc"
echo ""
echo "Step 2: Keep the terminal open, then open your browser and go to:"
echo ""
echo "    http://localhost:${PORT}"
echo ""
echo "Step 3: If asked for a token/password, verify it below in this log."
echo "========================================================================"

# 4. 启动 Jupyter Notebook
# --ip=0.0.0.0 允许外部连接 (通过隧道)
# --no-browser 禁止在服务器尝试打开浏览器
unset XDG_RUNTIME_DIR
jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0