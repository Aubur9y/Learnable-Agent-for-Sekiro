import neptune
import torch
import os
import argparse
from PPO_network import PPO_Agent
from ppo_w_transfer_learning import PPO_Agent_tl
from env import SekiroEnv

run = neptune.init_run(
    project="aubury/sekiro",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MTAyMTdhMS03YWRmLTQ4YzUtYTE5Zi0yYTU2OTQxNzVkM2QifQ==",
)

parser = argparse.ArgumentParser(description='Evaluate a trained ppo model (with transfer learning)')
parser.add_argument('input_actor_network_path', type=str, help='input actor network path')
parser.add_argument('input_critic_network_path', type=str, help='input critic network path')
parser.add_argument('--model_type', type=str, default='PPO', help='PPO or PPO_tl')
parser.add_argument('--episodes', type=int, default=100, help='number of episodes to run')

args = parser.parse_args()

num_eval_episodes = args.episodes
model_used = args.model_type

if model_used == 'PPO':
    agent = PPO_Agent(n_actions=7, input_channels=3, height=224, width=224, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=10, n_epoch=4)
elif model_used == 'DQN':
    agent = PPO_Agent_tl(n_actions=7, input_channels=3, height=224, width=224, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=10, n_epoch=4)
else:
    raise ValueError(f"Invalid model type: {model_used}")

if not os.path.isfile(args.input_actor_network__path) or not os.path.isfile(args.input_critic_network_path):
    print(f"Error: The input model file '{args.input_actor_network__path}' or '{args.input_critic_network_path}' does not exist.")
    exit(1)

agent.load_model(args.input_actor_network__path, args.input_critic_network_path)

env = SekiroEnv()

for episode in range(num_eval_episodes):
    state = env.reset()
    total_reward = []
    done = False
    time_defeat_boss = 0

    reshaped_state_0 = state[0].reshape(3, 224, 224)
    while not done:
        action, prob, val = agent.choose_action(reshaped_state_0)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward.append(reward)

    average_reward = sum(total_reward) / len(total_reward)
    run["ppo/evaluate/episode/average reward per episode"].append(average_reward)

win_ratio = env.get_boss_death_count() / env.get_player_death_count()
print('win ratio:', win_ratio)

run.stop()


