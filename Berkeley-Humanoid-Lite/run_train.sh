#!/bin/bash

# ==========================================
# 1. 路径定义 (请确保与你的物理路径一致)
# ==========================================
export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"
export ISAAC_LAB_PATH="/home/mryan2005/IsaacLab"
export PROJECT_ROOT="/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite"

# ==========================================
# 2. 构建 PYTHONPATH (只包含源码路径，避开 Conda 干扰)
# ==========================================
export PYTHONPATH=""
# (A) Berkeley Humanoid 项目源码
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_assets
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_lowlevel

# (B) Isaac Lab 源码
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/isaaclab
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_rl
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks

# ==========================================
# 3. 设置 5.1 必需的环境变量
# ==========================================
# 强制加载 Isaac Lab 的 5.1 配置文件
export ISAACLAB_EXP_FILE="$ISAAC_LAB_PATH/apps/isaaclab.python.kit"

# ==========================================
# 4. 执行训练启动 (使用 python.sh)
# ==========================================
cd $PROJECT_ROOT

echo "[INFO] 正在启动 Berkeley Humanoid Lite 训练..."
echo "[INFO] 显卡: RTX 4060 Laptop | 模式: Headless (推荐用于训练)"

# 参数说明:
# --task: 任务名称
# --num_envs: 环境数量 (RTX 4060 建议 2048-4096)
# --headless: 不显示 UI 窗口，训练速度提升 2-5 倍
# --video: 训练期间不建议开启，会极大地降低速度并消耗显存

$ISAACSIM_PATH/python.sh ./scripts/rsl_rl/train.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 2048 \
    "$@"