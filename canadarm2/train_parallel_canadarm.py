import gymnasium
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler

from skrl.resources.schedulers.torch import KLAdaptiveRL
from torch.optim.lr_scheduler import LinearLR


from skrl.trainers.torch.parallel import ParallelTrainer
from skrl.utils import set_seed
from gymnasium.vector import AsyncVectorEnv
import multiprocessing as mp

from canadarm_env.env_canadarm import CanadarmEnv

# Semilla
set_seed(42)

# Modelos
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"].to(self.device))), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# linear_lr_scheduler.py
class LinearLRScheduler:
    def __init__(self, optimizer, total_timesteps, initial_lr):
        self.optimizer = optimizer
        self.total_timesteps = total_timesteps
        self.initial_lr = initial_lr
        self.current_step = 0

    def __call__(self):
        # Paso actual como fracción del total
        progress = self.current_step / self.total_timesteps
        lr = self.initial_lr * (1.0 - progress)
        lr = max(lr, 1e-6)  # evita valores demasiado pequeños

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1


# Guardar el primer entorno real para acceder a métodos como save_all_reward_plots
first_env = None
def make_env(seed=None):
    def _init():
        global first_env
        env = CanadarmEnv()
        return env
    return _init

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    num_envs = 6
    envs = AsyncVectorEnv([make_env(i) for i in range(num_envs)])
    env = wrap_env(envs)  # <- esto sí es un entorno paralelo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Memoria con num_envs = 4
    memory = RandomMemory(memory_size=10000, num_envs=num_envs, device=device)

    # Modelos
    models = {
        "policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
        "value": Value(env.observation_space, env.action_space, device)
    }
    timesteps = 7000000
   # Configuración PPO
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 10000 #1024 2048 4096 8192
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.99
    cfg["learning_rate"] = 4.5e-4 
    cfg["learning_rate_scheduler"] = LinearLR
    cfg["learning_rate_scheduler_kwargs"] = {
        "start_factor": 1.0,
        "end_factor": 0.05,
        "total_iters": 5000
    }
    cfg["grad_norm_clip"] = 0.5
    cfg["ratio_clip"] = 0.2
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = timesteps / 2000
    cfg["experiment"]["checkpoint_interval"] = timesteps / 100
    cfg["experiment"]["directory"] = str(Path.home() / "TFG-Canadarm2/canadarm2/runs/torch/Canadarm")
    cfg["experiment"]["store_separately"] = False

    # Agente
    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # Entrenamiento paralelo con 4 entornos
    cfg_trainer = {"timesteps": timesteps, "headless": True}
    trainer = ParallelTrainer(env=env, agents=[agent], cfg=cfg_trainer)
    trainer.train()
