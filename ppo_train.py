import neptune
import torch
import argparse
import sys
import os
import time
import pygetwindow as gw
from PPO_network import PPO_Agent
from env import SekiroEnv

parser = argparse.ArgumentParser(
    usage='''python3 ppo_train.py model/{model name}.pth'''
)

parser.add_argument('model', type=str, help='model path')

args = parser.parse_args()

if os.path.isfile(args.model):
    print(f"The model '{args.model}' already exists.")
    sys.exit(1)

width = 224
height = 224
input_channels = 3
n_actions = 7
save_frequency = 5

# Constants for PPO
N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
EPISODES = 2
gamma = 0.99
gae_lambda = 0.95
policy_clip = 0.2

# run = neptune.init_run(
#     project="aubury/sekiro",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MTAyMTdhMS03YWRmLTQ4YzUtYTE5Zi0yYTU2OTQxNzVkM2QifQ==",
# )

def wait_for_sekiro_window():
    while True:
        sekiro_window = gw.getWindowsWithTitle('Sekiro')
        if sekiro_window:
            print("Found Sekiro window. Starting program...")
            sekiro_window[0].moveTo(-10, 0)  # position the window
            break
        else:
            print("Sekiro window not found. Waiting...")
            time.sleep(1)

if __name__ == '__main__':
    wait_for_sekiro_window()

    agent = PPO_Agent(n_actions, input_channels, height, width, gamma, alpha, gae_lambda, policy_clip, batch_size, N, n_epochs)

    env = SekiroEnv()

    n_steps = 0

    for episode in range(EPISODES):
        done = False
        state = env.reset()  # img, agent_hp, agent_ep, boss_hp
        reshaped_state_0 = state[0].reshape(input_channels, width, height)

        while not done:
            action, prob, val = agent.choose_action(reshaped_state_0)
            next_state, reward, done = env.step(action)

            # run["ppo/train/batch/reward"].append(reward)

            n_steps += 1
            agent.store_data(reshaped_state_0, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
            state = next_state

        if episode % save_frequency == 0:
            agent.save_model()

