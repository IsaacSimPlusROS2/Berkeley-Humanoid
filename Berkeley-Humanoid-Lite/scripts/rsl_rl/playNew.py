"""
使用 RSL-RL v1.0.2 推理/预览 Berkeley Humanoid Lite 的脚本。
结构严格仿照 train.py，适配 Isaac Sim 5.1。
"""

import argparse
import os
import sys
import torch

# --- [1] 初始化 AppLauncher (必须最先导入) ---
from isaaclab.app import AppLauncher

# 解决本地导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play/Inference with RSL-RL agent.")
parser.add_argument("--num_envs", type=int, default=1, help="仿真环境数量(预览通常为1)")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="随机种子")
# 添加 RSL-RL 参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 强制开启图形界面
args_cli.headless = False

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [2] 仿真启动后的导入 ---
import gymnasium as gym
from omegaconf import OmegaConf

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.dict import class_to_dict

import berkeley_humanoid_lite.tasks  # noqa: F401

# --- [兼容层] 复用 Wrapper ---
class RslRlVecEnvWrapperFixed(RslRlVecEnvWrapper):
    def __init__(self, env):
        self.env = env
        base_env = env.unwrapped
        self.unwrapped_env = base_env 
        self.device = base_env.device
        
        obs_manager = base_env.observation_manager
        
        def _to_int(val):
            if isinstance(val, (tuple, list)): return int(val[0])
            return int(val)

        self.num_obs = _to_int(obs_manager.group_obs_dim["policy"])
        self.num_actions = _to_int(base_env.action_manager.total_action_dim)
        self.num_envs = base_env.num_envs

        if "critic" in obs_manager.group_obs_dim:
            self.num_privileged_obs = _to_int(obs_manager.group_obs_dim["critic"])
        else:
            self.num_privileged_obs = None
            
        self.episode_length_buf = base_env.episode_length_buf

    def _strip_tensordict(self, obs):
        if obs is None: return None
        if not isinstance(obs, torch.Tensor):
            if isinstance(obs, dict):
                return torch.cat(list(obs.values()), dim=-1)
        return obs.view(obs.shape)

    def get_observations(self):
        obs = self.unwrapped.observation_manager.compute_group("policy")
        return self._strip_tensordict(obs)

    def get_privileged_observations(self):
        if self.num_privileged_obs is not None:
            obs = self.unwrapped.observation_manager.compute_group("critic")
            return self._strip_tensordict(obs)
        return None

    def step(self, actions):
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = self._strip_tensordict(obs_dict["policy"])
        privileged_obs = None
        if "critic" in obs_dict:
            privileged_obs = self._strip_tensordict(obs_dict["critic"])
        elif self.num_privileged_obs is not None:
            privileged_obs = self.get_privileged_observations()
        dones = terminated | truncated
        return obs, privileged_obs, rew, dones, extras
    
    def reset(self):
        obs_dict, _ = self.env.reset()
        obs = self._strip_tensordict(obs_dict["policy"])
        return obs, self.get_privileged_observations()

# --- [参数过滤器] ---
def filter_dict(raw_dict, whitelist):
    return {k: v for k, v in raw_dict.items() if k in whitelist}

PPO_WHITELIST = ['value_loss_coef', 'use_clipped_value_loss', 'clip_param', 'entropy_coef', 
                 'num_learning_epochs', 'num_mini_batches', 'learning_rate', 'schedule', 
                 'gamma', 'lam', 'desired_kl', 'max_grad_norm']
POLICY_WHITELIST = ['init_noise_std', 'actor_hidden_dims', 'critic_hidden_dims', 'activation']

# -------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # 1. 同步参数
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # === 地形与高度修正 ===
    if hasattr(env_cfg.scene, "terrain"):
        print("[INFO] Play模式: 强制地形为无限平面 (Plane)")
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None 
    
    if hasattr(env_cfg.scene, "robot"):
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0)
        print("[INFO] Play模式: 强制初始高度为 0m")

    # 2. 设置日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    # 3. 寻找 Checkpoint
    load_run = agent_cfg.load_run
    if load_run == "-1":
        load_run = None
    
    resume_path = None
    try:
        resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] 加载模型路径: {resume_path}")
    except Exception as e:
        print(f"[ERROR] 无法找到模型 checkpoint: {e}")
        simulation_app.close()
        sys.exit(1)

    # 4. 创建环境
    print(f"[INFO] 正在创建环境: {args_cli.task}")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    
    env = RslRlVecEnvWrapperFixed(env)

    # 5. 构造 Runner 配置
    raw_dict = class_to_dict(agent_cfg) if not isinstance(agent_cfg, dict) else agent_cfg
    
    # === [关键修复] 补全 runner 所需的所有键值 ===
    rsl_cfg = {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "experiment_name": agent_cfg.experiment_name,
            "checkpoint": resume_path,
            
            # --- 以下是本次修复补充的必要参数 ---
            "num_steps_per_env": agent_cfg.num_steps_per_env,  # 解决 KeyError
            "max_iterations": agent_cfg.max_iterations,        # 初始化需要
            "save_interval": agent_cfg.save_interval,          # 初始化需要
            "run_name": agent_cfg.run_name,
        },
        "algorithm": filter_dict(raw_dict.get("algorithm", {}), PPO_WHITELIST),
        "policy": filter_dict(raw_dict.get("policy", {}), POLICY_WHITELIST),
    }

    # 6. 初始化 Runner
    # log_dir=None 表示不创建新的日志文件夹
    runner = OnPolicyRunner(env, rsl_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    
    policy = runner.get_inference_policy(device=env.device)

    # 7. 推理循环
    print("-" * 80)
    print("[INFO] 启动成功！在 Isaac Sim 中按 'F' 键跟随机器人。")
    print("-" * 80)

    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()