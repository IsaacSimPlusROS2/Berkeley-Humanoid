#!/bin/bash

# ==========================================
# 1. 基础路径 (请确认无误)
# ==========================================
export ISAACSIM_PATH="/home/mryan2005/isaac-sim-standalone-5.1.0-linux-x86_64"
export ISAAC_LAB_PATH="/home/mryan2005/IsaacLab"
export PROJECT_ROOT="/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite"

# ==========================================
# 2. [核心] 彻底清除环境变量污染
# ==========================================
# 清除所有可能干扰 Isaac Sim 5.1 的 Python 变量
unset PYTHONPATH
unset CONDA_PREFIX
unset PYTHONHOME

# ==========================================
# 3. [核心] 重新构建极简 PYTHONPATH
# ==========================================
# 我们只添加源码目录，绝对不添加任何包含 'exts' 或 'extscache' 的路径
# 让 python.sh 自己去处理那些二进制库
export PYTHONPATH=$PROJECT_ROOT/source/berkeley_humanoid_lite
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/source/berkeley_humanoid_lite_assets
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/isaaclab
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_rl
export PYTHONPATH=$PYTHONPATH:$ISAAC_LAB_PATH/source/extensions/omni.isaac.lab_tasks

# ==========================================
# 4. [关键] 强制使用 Isaac Sim 原生配置
# ==========================================
# 使用内置的 isaacsim.python.kit 而不是 isaaclab 的 kit
# 这会防止 SimulationApp 自动去加载那些会导致 "ObjectType" 重复注册的扩展
export ISAACLAB_EXP_FILE="$ISAACSIM_PATH/apps/isaacsim.python.kit"

# ==========================================
# 5. 执行启动
# ==========================================
cd $PROJECT_ROOT

echo "[INFO] 正在启动 Isaac Sim 5.1 预览 (纯净模式)..."

# 执行预览 (num_envs=1)
$ISAACSIM_PATH/python.sh ./scripts/rsl_rl/playNew.py \
    --task Velocity-Berkeley-Humanoid-Lite-v0 \
    --num_envs 1 \
    --load_run 2026-02-10_15-54-39