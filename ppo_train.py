import neptune
import torch
import argparse
import sys
import os
import time
import pygetwindow as gw
from PPO_network import PPO_Agent
from ppo_w_transfer_learning import PPO_Agent_tl
from env import SekiroEnv
import keyboard

def wait_for_start_key():
    print("Press 'T' to start training.")
    keyboard.wait('t')  # This blocks until 'T' is pressed
    print("Training started.")

def check_for_stop_key():
    if keyboard.is_pressed('t'):  # Check if 'T' is pressed
        print("Training stopped by user.")
        return True
    return False

parser = argparse.ArgumentParser(
    usage='''python3 ppo_train.py model/{model name}.pth'''
)

parser.add_argument('actor_checkpoint', type=str, help='actor checkpoint path')
parser.add_argument('critic_checkpoint', type=str, help='critic checkpoint path')
parser.add_argument('EPISODES', type=int, help='number of episodes')

args = parser.parse_args()

if os.path.isfile(args.actor_checkpoint) or os.path.isfile(args.critic_checkpoint):
    print(f"The model '{args.actor_checkpoint}' or '{args.critic_checkpoint}' already exists.")
    sys.exit(1)

width = 224
height = 224
input_channels = 3
n_actions = 7
save_frequency = 5

transfer_learning_enabled = False

file_path_actor = args.actor_checkpoint
file_path_critic = args.critic_checkpoint

# Constants for PPO
N = 10
batch_size = 16  # 32-64
n_epochs = 4  # 4-10
alpha = 0.0003  # 0.0001-0.0003
EPISODES = args.EPISODES
gamma = 0.99  # 0.98-0.99
gae_lambda = 0.95  # 0.92-0.98
policy_clip = 0.2  # 0.1-0.2

run = neptune.init_run(
    project="aubury/sekiro",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MTAyMTdhMS03YWRmLTQ4YzUtYTE5Zi0yYTU2OTQxNzVkM2QifQ==",
)

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
    wait_for_start_key()

    if transfer_learning_enabled:
        agent = PPO_Agent_tl(n_actions, input_channels, height, width, gamma, alpha, gae_lambda, policy_clip, batch_size, N, n_epochs)
    else:
        agent = PPO_Agent(n_actions, input_channels, height, width, gamma, alpha, gae_lambda, policy_clip, batch_size, N, n_epochs)

    env = SekiroEnv()

    n_steps = 0

    for episode in range(EPISODES):
        total_reward = []
        done = False
        state = env.reset()  # img, agent_hp, agent_ep, boss_hp
        reshaped_state_0 = state[0].reshape(input_channels, width, height)

        # normalized_image = reshaped_state_0 / 255.0

        while not done:
            action, prob, val = agent.choose_action(reshaped_state_0)
            next_state, reward, done = env.step(action)

            total_reward.append(reward)

            run["ppo/train/batch/reward"].append(reward)

            n_steps += 1
            agent.store_data(reshaped_state_0, action, prob, val, reward, done)
            if n_steps % N == 0:
                loss = agent.learn()
                run["ppo/train/batch/loss"].append(loss)
            state = next_state

            if check_for_stop_key():
                agent.save_model(file_path_actor, file_path_critic)
                run.stop()
                sys.exit(0)

        if episode % save_frequency == 0:
            agent.save_model(file_path_actor, file_path_critic)

        average_reward_per_episode = sum(total_reward) / len(total_reward)

        run["ppo/train/episode/average reward per episode"].append(average_reward_per_episode)

    print(f"Training finished, a total of {EPISODES} episodes and {n_steps} steps.")



