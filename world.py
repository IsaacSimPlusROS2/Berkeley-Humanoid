from isaacsim import SimulationApp

# 1. 启动仿真
simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

# ================= 配置区域 =================
USD_PATH = "/home/mryan2005/berkeley_humanoid_isaac/Berkeley-Humanoid-Lite-Assets/data/robots/berkeley_humanoid/berkeley_humanoid_lite/usd/berkeley_humanoid_lite.usd"
ROBOT_PRIM_PATH = "/World/BerkeleyHumanoid"

# 这是一个统一的关节列表，不分部位，放在一起处理
JOINT_NAMES = [
    "arm_left_shoulder_pitch_joint", "arm_left_shoulder_roll_joint", "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint", "arm_left_elbow_roll_joint",
    "arm_right_shoulder_pitch_joint", "arm_right_shoulder_roll_joint", "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint", "arm_right_elbow_roll_joint",
    "leg_left_hip_roll_joint", "leg_left_hip_yaw_joint", "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint", "leg_left_ankle_pitch_joint", "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint", "leg_right_hip_yaw_joint", "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint", "leg_right_ankle_pitch_joint", "leg_right_ankle_roll_joint"
]

# ================= 场景构建 =================
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM_PATH)

# 【核心设置】：Base (骨盆) 的位置
# 我们把机器人作为一个整体加载。
# 必须把 Z 设为 0.72 (约72厘米)，这是为了让脚掌刚好接触地面 (配合膝盖弯曲)
# 如果 Z=0，腰就在地上，机器人必倒。
my_humanoid = world.scene.add(
    SingleArticulation(
        prim_path=ROBOT_PRIM_PATH,
        name="humanoid",
        position=np.array([0.0, 0.0, 0.025]) 
    )
)

# ================= 初始化物理 =================
world.reset()

# 建立名称到索引的映射
dof_map = {}
for i, name in enumerate(JOINT_NAMES):
    idx = my_humanoid.get_dof_index(name)
    if idx is not None:
        dof_map[name] = idx

# ================= 统一设置刚度 (PD Control) =================
# 机器人要站稳，全身都需要力气。这里统一设置，不再硬拆。
controller = my_humanoid.get_articulation_controller()
num_dofs = my_humanoid.num_dof

# 默认全身刚度高 (800)，阻尼中等 (40) -> 保证站立不软腿
kps = np.full(num_dofs, 800.0) 
kds = np.full(num_dofs, 40.0)

# 针对手臂稍微调软一点，让挥手更顺滑 (可选，不拆也可以，这里为了动作自然微调)
for name, idx in dof_map.items():
    if "arm" in name:
        kps[idx] = 200.0
        kds[idx] = 10.0

controller.set_gains(kps=kps, kds=kds)

# ================= 仿真主循环 =================
print("仿真开始。点击 Play，机器人将保持下蹲平衡并挥手。")

while simulation_app.is_running():
    world.step(render=True)
    
    if world.is_playing() and my_humanoid.handles_initialized:
        t = world.current_time
        
        # 创建一个全零的目标数组
        target_positions = np.zeros(num_dofs)
        
        # 【统一逻辑】：在一个循环里处理全身所有关节
        # 不需要把 list 拆成两半，我们直接根据关节名字来决定动作
        for name, idx in dof_map.items():
            
            # --- 1. 平衡逻辑 (下半身) ---
            # 采用“半蹲”姿态，这是最稳的：髋部后坐，膝盖前顶，脚踝回勾
            if "hip_pitch" in name:
                target_positions[idx] = -0.3  # 髋部弯曲 (屁股向后)
            elif "knee_pitch" in name:
                target_positions[idx] = 0.6   # 膝盖弯曲 (降低重心)
            elif "ankle_pitch" in name:
                target_positions[idx] = -0.3  # 脚踝背屈 (保持脚掌平地)
            
            # --- 2. 挥手逻辑 (上半身) ---
            elif "shoulder_pitch" in name:
                # 只有肩部 Pitch 关节做正弦波运动
                target_positions[idx] = 1.0 * np.sin(t * 3.0)
            elif "elbow_pitch" in name:
                # 肘部配合弯曲
                target_positions[idx] = 0.5 * np.cos(t * 3.0)
            
            # --- 3. 其他关节 ---
            else:
                target_positions[idx] = 0.0   # 保持归零

        # 应用全身动作
        my_humanoid.apply_action(ArticulationAction(joint_positions=target_positions))

simulation_app.close()