import gymnasium as gym

import torch

# import the skrl components to build the RL system
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, TabularMixin
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.resources.preprocessors.torch import RunningStandardScaler

# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed


# define model (tabular model) using mixin
class EpilonGreedyPolicy(TabularMixin, Model):
    def __init__(self, observation_space, action_space, device, num_envs=1, epsilon=0.1):
        Model.__init__(self, observation_space, action_space, device)
        TabularMixin.__init__(self, num_envs)

        self.epsilon = 0
        self.q_table = torch.ones((num_envs, self.num_observations, self.num_actions),
                                  dtype=torch.float32, device=self.device)

    def compute(self, inputs, role):
        actions = torch.argmax(self.q_table[torch.arange(self.num_envs).view(-1, 1), 
                                    inputs["states"].long()],  # <- Conversión a long
                       dim=-1, keepdim=True).view(-1, 1)

        # choose random actions for exploration according to epsilon
        indexes = (torch.rand(inputs["states"].shape[0], device=self.device) < self.epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.randint(self.num_actions, (indexes.numel(), 1), device=self.device)
        return actions, {}


# load and wrap the gymnasium environment.
# note: the environment version may change depending on the gymnasium version
try:
    env = gym.make("FrozenLake-v1", is_slippery=False,render_mode="rgb_array")
except (gym.error.DeprecatedEnv, gym.error.VersionNotFound) as e:
    env_id = [spec for spec in gym.envs.registry if spec.startswith("FrozenLake-v")][0]
    print("FrozenLake-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# instantiate the agent's model (table)
# Q-learning requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#models
models = {}
models["policy"] = EpilonGreedyPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, epsilon=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/q_learning.html#configuration-and-hyperparameters
cfg = Q_LEARNING_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.999
cfg["alpha"] = 0.3
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "runs/torch/FrozenLake"

agent = Q_LEARNING(models=models,
                   memory=None,
                   cfg=cfg,
                   observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=device)


#=================== manual interaction with the environment =================

# load checkpoints
# state_preprocessor.load_state_dict(torch.load("state_preprocessor_1600.pt"))
agent.load("./best_agent.pt")

# Configurar el entorno para evaluación
eval_env = wrap_env(env)

# Establecer el agente en modo de evaluación
agent.set_mode("eval")

num_episodios = 100
exitos = 0
time_steps = 0
for ep in range(num_episodios):
    states, _ = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action = agent.policy.act({"states": states})[0]

        next_states, rewards, terminated, truncated, infos = env.step(action)
        done = terminated.any() or truncated.any()
        states = next_states
        time_steps += 1
        if terminated.any() and rewards.item() == 1.0:
            exitos += 1

print(f"\n✅ Éxitos: {exitos}/{num_episodios} ({(exitos / num_episodios) * 100:.1f}%)")


env.close()
