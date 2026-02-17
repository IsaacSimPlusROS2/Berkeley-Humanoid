# scripts/rsl_rl/train.py

"""使用 RSL-RL v1.0.2 训练 Berkeley Humanoid Lite 的脚本。适配 Isaac Sim 5.1。"""

import argparse
import os
import sys
import pickle

# --- [关键] 1. 初始化 AppLauncher ---
from isaaclab.app import AppLauncher

# 解决本地导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=None, help="仿真环境数量")
parser.add_argument("--task", type=str, default=None, help="任务名称")
parser.add_argument("--seed", type=int, default=None, help="随机种子")
parser.add_argument("--max_iterations", type=int, default=None, help="最大训练迭代数")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- [关键] 2. 仿真启动后的导入 ---
import gymnasium as gym
import torch
from datetime import datetime
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

# --- [兼容层] 适配 Isaac Lab 0.54 和 RSL-RL v1.0.2 ---
class RslRlVecEnvWrapperFixed(RslRlVecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        base_env = env.unwrapped
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
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 2. 设置日志根目录
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    # 3. 处理接续训练路径 (核心修复点)
    load_run = None
    if agent_cfg.resume:
        load_run = str(agent_cfg.load_run)
        # 如果用户传入了包含路径的字符串，只保留最后一部分（文件夹名）
        if "/" in load_run:
            load_run = os.path.basename(load_run.strip("/"))
        # 如果是 -1，设为 None 以便 get_checkpoint_path 寻找最新
        if load_run == "-1":
            load_run = None
            
        print(f"[INFO] 正在搜索接续目录，根路径: {log_root_path}, 目标: {load_run if load_run else '最新'}")

    # 4. 设置本次训练的日志目录
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    os.makedirs(log_dir, exist_ok=True)

    # 5. 创建环境
    print(f"[INFO] 正在创建环境: {args_cli.task} (数量: {env_cfg.scene.num_envs})")
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapperFixed(env)

    # 6. 构造 RSL-RL v1.0.2 字典
    raw_dict = agent_cfg.to_dict()
    rsl_cfg = {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": agent_cfg.num_steps_per_env,
            "max_iterations": agent_cfg.max_iterations,
            "save_interval": agent_cfg.save_interval,
            "experiment_name": agent_cfg.experiment_name,
            "run_name": agent_cfg.run_name,
            "resume": agent_cfg.resume,
            "load_run": agent_cfg.load_run,
            "load_checkpoint": agent_cfg.load_checkpoint,
            "checkpoint": "model_*.pt",
        },
        "algorithm": filter_dict(raw_dict.get("algorithm", {}), PPO_WHITELIST),
        "policy": filter_dict(raw_dict.get("policy", {}), POLICY_WHITELIST),
    }

    # 7. 创建 Runner
    runner = OnPolicyRunner(env, rsl_cfg, log_dir=log_dir, device=agent_cfg.device)
    
    # 8. 加载权重 (修复后的调用)
    if agent_cfg.resume:
        try:
            resume_path = get_checkpoint_path(log_root_path, load_run, agent_cfg.load_checkpoint)
            print(f"[INFO] 成功定位模型: {resume_path}")
            runner.load(resume_path)
        except Exception as e:
            print(f"[ERROR] 恢复模型失败: {e}")
            simulation_app.close()
            sys.exit(1)

    # 9. 保存配置备份
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    try:
        env_dict = class_to_dict(env_cfg)
        with open(os.path.join(params_dir, "env.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(env_dict))
    except:
        pass
        
    # 10. 开始训练
    print(f"[INFO] 训练开始。目标迭代次数: {agent_cfg.max_iterations}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()