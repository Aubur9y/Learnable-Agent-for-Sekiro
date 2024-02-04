import torch
import os
import argparse
from DQN_network import DQN_Agent
from DuellingDQN_network import DeullingDQN_Agent
from env import SekiroEnv

parser = argparse.ArgumentParser(description='Evaluate a trained DQN or Duelling DQN model')
parser.add_argument('input_model_path', type=str, help='input model path')
parser.add_argument('--model_type', type=str, default='DQN', help='DQN or Duelling_DQN')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes to run')

args = parser.parse_args()

num_eval_episodes = args.episodes
model_used = args.model_type

if model_used == 'Duelling_DQN':
    agent = DeullingDQN_Agent(n_actions=7, input_channels=3, height=224, width=224, batch_size=64, gamma=0.99, lr=0.0003, epsilon=1)
elif model_used == 'DQN':
    agent = DQN_Agent(n_actions=7, input_channels=3, height=224, width=224, batch_size=64, gamma=0.99, lr=0.0003, epsilon=1)
else:
    raise ValueError(f"Invalid model type: {model_used}")

if not os.path.isfile(args.input_model_path):
    print(f"Error: The input model file '{args.input_model_path}' does not exist.")
    exit(1)

agent.load_model(args.input_model_path)

env = SekiroEnv()

total_rewards = []

for episode in range(num_eval_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    time_defeat_boss = 0

    reshaped_state_0 = state[0].reshape(3, 224, 224)
    while not done:
        action, _ = agent.choose_action(reshaped_state_0)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

    total_rewards.append(total_reward)
    print(f"Episode {episode + 1}: {total_reward}")

average_reward = sum(total_rewards) / len(total_rewards)
win_ratio = env.get_boss_death_count() / env.get_player_death_count()


