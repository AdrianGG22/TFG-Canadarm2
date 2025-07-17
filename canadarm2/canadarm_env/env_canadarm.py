from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from mujoco import MjModel, MjData
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os

class CanadarmEnv(MujocoEnv, utils.EzPickle):
    metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": 50,
    }
    def __init__(self,
                 model_path= str(Path.home() / "TFG-Canadarm2/canadarm2/canadarm_env/models/SSRMS_Canadarm2.xml"),
                 frame_skip: int = 10,
                 max_time: int = 45,
                 target_position=np.array([8.5, 3, 9]),
                 graficas_tsboard: bool = True,
                 seed: int = None,
                 **kwargs):
        
        utils.EzPickle.__init__(self, **kwargs)

        self.frame_skip = frame_skip

        # 7 pos articulares
        # 7 vel articulares
        # 7 acc articulares
        # 3 pos EE
        # 3 pos target 
        # 7 acción actual
        # 7 acción anterior

        observation_space = Box(low=-np.inf, high=np.inf, shape=(41,), dtype=np.float64)
        
        MujocoEnv.__init__(
            self, model_path, self.frame_skip, observation_space=observation_space, **kwargs
        )  

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")

        self.action_space = Box(low=-(1.5 * np.pi), high=(1.5 * np.pi), shape=(7,), dtype=np.float64)

        self.init_qpos = np.array([-1.13, 0.0, 0., -1.32, 0.0, 0.0, 0.0])
        self.init_qvel = np.zeros_like(self.init_qpos)
        
        self.target_position = target_position
        self.dift = self.model.opt.timestep * self.frame_skip
        self.max_steps = max_time / self.dift
        self.graficas_tsboard = graficas_tsboard


        self.current_step = 0
        self.prev_action = np.zeros(7, dtype=np.float64)
        self.prev_qvel = None
        self.prev_accel = np.zeros(7, dtype=np.float64)
        self.last_action = np.zeros(7, dtype=np.float64)
        self.veces_terminadas = 0

        self.reward_logs = {
            "tracking": [],
            "reached": [],
            "vel_penalty": [],
            "accel_penalty": [],
            "torque_penalty": [],
            "action_penalty": [],
            "dist": [],
            "step": [],
        }

        self.writer = SummaryWriter(log_dir=str(Path.home() / "TFG-Canadarm2/canadarm2/runs/torch/Canadarm/env_logs"))
        self._step_count_tb = 0  # contador separado para logs

        # Establecer la semilla si se proporciona
        if seed is not None:
            self.seed(42)

    def step(self, action):
        
        truncated = False
        terminated = False
        
        # Procesamiento de la acción
        current_qpos = self.data.qpos.ravel().copy()  
        delta_q = 0.3* action
        processed_action = np.clip(current_qpos + delta_q, -(1.5 * np.pi), (1.5 * np.pi))

        # Cálculo de la aceleración
        qvel = self.data.qvel.ravel().copy()
        
        if self.prev_qvel is None:
            accel = np.zeros_like(qvel)
        else:
            accel = (qvel - self.prev_qvel) / (self.frame_skip * self.model.opt.timestep)
        self.prev_qvel = qvel.copy()
        self.prev_accel = accel.copy()

        self.prev_action = self.last_action.copy()
        self.last_action = processed_action.copy()

        # Se realiza un step de simulación
        self.do_simulation(processed_action, self.frame_skip)
        self.current_step += 1

        ee_pos = self.data.site_xpos[self.ee_site_id].copy()
        
        dist = np.linalg.norm(ee_pos - self.target_position)

        # Incializar variables de recompensa
        reached_rew = 0.0
        step_max_penalty = 0.0

        # RECOMPENSAS
        sigma = 0.7
        tracking_reward = 10 * np.exp(-dist / sigma)

        # Recompensa por alcanzar al objetivo
        if dist <= 0.1:
            reached_rew = 20 

        # Penalización por velocidades articulares
        vel_penalty = -5e-2 * np.linalg.norm(self.prev_qvel)

        # Penalización por aceleraciones articulares
        accel_penalty = -5e-5 * np.linalg.norm(self.prev_accel)

        # Penalizacion por torques
        torques = self.data.actuator_force.copy()
        torque_penalty = -1e-6 * np.sum(np.square(torques))

        # Penalización por cambios bruscos de acción
        act_penalty = -8e-3 * np.linalg.norm(self.last_action - self.prev_action)


        # TERMINACIONES

        # if dist <= 0.1:
        #     terminated = True

        # Por limite de pasos
        if self.current_step >= self.max_steps:
            step_max_penalty = -1
            truncated = True
        
        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        info = {
            "target": self.target_position,
            "ee_pos": ee_pos,
            "distance": dist,
            "step": self.current_step
        }

        reward = tracking_reward + reached_rew + vel_penalty + accel_penalty + torque_penalty + act_penalty + step_max_penalty

        # Guardar recompensas para análisis posterior
        self.reward_logs["tracking"].append(tracking_reward)
        self.reward_logs["reached"].append(reached_rew)
        self.reward_logs["vel_penalty"].append(vel_penalty)
        self.reward_logs["accel_penalty"].append(accel_penalty)
        self.reward_logs["torque_penalty"].append(torque_penalty)
        self.reward_logs["action_penalty"].append(act_penalty)
        self.reward_logs["dist"].append(dist)
        self.reward_logs["step"].append(self.current_step)

        self._step_count_tb += 1

        # Guardar en TensorBoard cada N pasos
        if (self._step_count_tb % 2000 == 0 or dist <= 0.1) and self.graficas_tsboard:
            for key, values in self.reward_logs.items():
                if key == "step":
                    continue
                if values:  # asegura que no esté vacío
                    self.writer.add_scalar(f"env/{key}", values[-1], self._step_count_tb)
        
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Obtener posicion del efector final
        ee_pos = self.data.site_xpos[self.ee_site_id].copy()

        direction_to_target = self.target_position - ee_pos
        direction_to_target /= np.linalg.norm(direction_to_target) + 1e-8  # evitar división por cero


        return np.concatenate([
            self.data.qpos.ravel().copy(),         # 7 pos articulares
            self.data.qvel.ravel().copy(),         # 7 vel articulares
            self.prev_accel.copy(),                # 7 acc articulares
            ee_pos.copy(),                         # 3 pos EE
            self.target_position.copy(),           # 3 pos target
            self.last_action.copy(),               # 7 acción actual
            self.prev_action.copy(),               # 7 acción anterior
        ]).astype(np.float64)

    def reset(self, *, seed=None, options=None):

        if seed is not None:
            self.seed(seed)
        
        # Reiniciar estado
        qpos = self.init_qpos 

        qvel = self.init_qvel

        self.set_state(qpos, qvel)

        self.current_step = 0
        self.prev_action = np.zeros(7, dtype=np.float64)
        self.last_action = np.zeros(7, dtype=np.float64)
        self.prev_qvel = None
        self.prev_accel = np.zeros(7, dtype=np.float64)

        return self._get_obs(), {}

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.5
