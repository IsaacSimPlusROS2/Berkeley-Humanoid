# scripts/rsl_rl/play.py

"""
使用 RSL-RL 运行 Berkeley Humanoid Lite 推理的脚本。
适配：Isaac Sim 5.1 / Isaac Lab 0.54+
"""

import argparse
import os
import sys

# --- [关键] 1. 初始化 AppLauncher (必须最先导入) ---
from isaaclab.app import AppLauncher

# 将当前目录加入路径以便导入 cli_args.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args

# 解析参数
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="是否录制视频")
parser.add_argument("--video_length", type=int, default=200, help="录制视频步数")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="禁用 Fabric")
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量")
parser.add_argument("--task", type=str, default=None, help="任务名称")

# 添加 RSL-RL 和 AppLauncher 的参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动仿真 App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [关键] 2. 仿真启动后的导入 ---
import gymnasium as gym
import torch
from omegaconf import OmegaConf

from rsl_rl.runners import OnPolicyRunner

import isaaclab.utils.string as string_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# 导入自定义任务包以注册环境
import berkeley_humanoid_lite.tasks  # noqa: F401

def main():
    """推理主函数"""
    
    # 1. 解析环境配置
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs if args_cli.num_envs is not None else 16, 
        use_fabric=not args_cli.disable_fabric
    )

    # 2. 解析 RSL-RL Agent 配置
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # 3. 寻找模型路径
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    resume_path = None
    try:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] 找到模型文件: {resume_path}")
    except Exception as e:
        print(f"[WARN] 未能自动找到模型: {e}")
        fallback_path = os.path.join("checkpoints", "policy_humanoid.pt")
        if os.path.exists(fallback_path):
            resume_path = fallback_path
            print(f"[INFO] 使用备用模型: {resume_path}")
        else:
            print("[INFO] 警告：未找到模型权重，进入“零动作测试模式”。")

    # 4. 创建环境
    render_mode = "rgb_array" if args_cli.video else None
    print(f"[INFO] 正在创建环境: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # 5. 录制视频包装
    if args_cli.video and resume_path:
        log_dir = os.path.dirname(resume_path)
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 6. 环境包装
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    # RslRlVecEnvWrapper 会规范化输入输出，供 RSL-RL 使用
    env = RslRlVecEnvWrapper(env)

    # --- 获取精确的维度信息 ---
    num_envs = env.num_envs
    num_actions = env.num_actions
    device = env.unwrapped.device

    # 7. 策略实例化
    if resume_path:
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=device)
    else:
        print(f"[INFO] 运行零动作策略 (维度: {num_envs} envs x {num_actions} actions)...")
        # 修复点：显式使用正确的维度 (num_envs, num_actions)
        def policy(obs):
            return torch.zeros((num_envs, num_actions), device=device)

    # 8. 仿真循环
    print(f"[INFO] 启动成功，正在运行...")
    
    # 获取初始观测 (RslRlVecEnvWrapper 返回单个 Tensor)
    obs = env.get_observations()
    
    while simulation_app.is_running():
        with torch.inference_mode():
            # 策略推理
            actions = policy(obs)
            # 步进：返回 obs, rewards, terminations, extras
            obs, rewards, terminations, extras = env.step(actions)

    # 9. 资源清理
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()