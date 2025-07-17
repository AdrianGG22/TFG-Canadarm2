import gymnasium as gym

import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch.optim.lr_scheduler import LinearLR
from pathlib import Path
from skrl.utils import set_seed

# Importar el entorno personalizado
from canadarm_env.env_canadarm import CanadarmEnv

set_seed(42)


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=device))


    def compute(self, inputs, role):

        return 2 * torch.tanh(self.net(inputs["states"].to(self.device))), self.log_std_parameter, {}

    
        
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

env = gym.make("Canadarm-v0", render_mode="rgb_array")
env = wrap_env(env)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


memory = RandomMemory(memory_size=10000, num_envs=6, device=device)

# Modelos
models = {
    "policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
    "value": Value(env.observation_space, env.action_space, device)
}

timesteps = 7_000_000

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
cfg["experiment"]["directory"] = "runs/torch/Canadarm"
cfg["experiment"]["store_separately"] = False

# Agente
agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# load agent

agent.load(str(Path.home() / "TFG-Canadarm2/canadarm2/trained_models/best_agent297.pt"))  

# ============= EVALUACIÓN CUANTITATIVA ================
NUM_EPISODES = 1
DISTANCE_THRESHOLD = 0.1

total_reward = 0
successful_episodes = 0
timestep = 0
print("\n=== EVALUACIÓN CUANTITATIVA ===")
for ep in range(NUM_EPISODES):
    states, infos = env.reset()
    episode_reward = 0
    reached_target = False

    for _ in range(1_000_000):
        with torch.no_grad():
            actions = agent.act(states=states, timestep=0, timesteps=0)[0]

        next_states, rewards, terminated, truncated, infos = env.step(actions)
        episode_reward += rewards.item()

        if infos.get('distance') <= DISTANCE_THRESHOLD:
            reached_target = True
            break

        if terminated or truncated:
            break
        states = next_states

    total_reward += episode_reward
    if reached_target:
        successful_episodes += 1

    print(f"[{ep+1}/{NUM_EPISODES}] Recompensa: {episode_reward:.2f} | Objetivo alcanzado: {reached_target}")

mean_reward = total_reward / NUM_EPISODES
print("\n==== RESULTADOS ====")
print(f"Recompensa media: {mean_reward:.2f}")
print(f"Éxitos (objetivo alcanzado): {successful_episodes}/{NUM_EPISODES}")



# ============= EVALUACIÓN CUALITATIVA ================

env = gym.make("Canadarm-v0", render_mode="human")
env = wrap_env(env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n=== EVALUACIÓN CUALITATIVA (visualización del movimiento) ===")

states, infos = env.reset()
for _ in range(1_000_000):
    with torch.no_grad():
        actions = agent.act(states=states, timestep=0, timesteps=0)[0]

    next_states, rewards, terminated, truncated, infos = env.step(actions)

    env.render()
    print(f"Distance: {infos.get('distance'):.3f}, \nEE_POS: {infos.get('ee_pos')}, \nTARGET: {infos.get('target')}")

    if infos.get("distance", 999) <= DISTANCE_THRESHOLD:
        print("Objetivo alcanzado!")
        break
    
    if terminated or truncated:
        states, infos = env.reset()

        

    states = next_states

env.close()