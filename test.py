import json
import sys
import logging
import torch
import argparse
import keyboardAction
import time
import keyboard
import pygetwindow as gw
import matplotlib.pyplot as plt
import os
from DQN_network import Agent
from env import SekiroEnv
from icecream import ic

# parser = argparse.ArgumentParser(
#     usage='''python3 train.py model/{model name}.pth'''
# )
# parser.add_argument('model', type=str, help='model path')
#
# # parse the arguments
# args = parser.parse_args()
#
# if os.path.isfile(args.model):
#     print(f"The model '{args.model}' already exists.")
#     sys.exit(1)

# Constants
width = 128
height = 128
input_channels = 1
battle_area = (360, 180, 300, 320)
boss_blood_area = (59, 90, 212, 475)
player_blood_area = (53, 560, 305, 5)
EPISODES = 2
episode_count = 0
input_dims = (input_channels, height, width)
n_actions = 7
batch_size = 16
gamma = 0.99

base_model_name = 'model/'
# experimenting learning rate
learning_rates = [0.001, 0.003, 0.005, 0.01]

model_names = {lr: f"{base_model_name}_lr_{lr}.pth" for lr in learning_rates}


epsilon = 1.0
save_frequency = 5
paused = True
episode_numbers = []
average_rewards = []
# file_path = args.model

def check_pause(paused_flag):
    """
    Toggles the paused flag when 'p' is pressed.
    Returns the updated paused state.
    """
    if keyboard.is_pressed('p'):
        paused_flag = not paused_flag  # Toggle the paused state
        print("Game is {}".format("paused" if paused_flag else "starting"))
        print("Press 'p' to {}".format("continue" if paused_flag else "pause"))
        time.sleep(2)  # has to wait for current motion to end in order to pause
        # Wait for the 'p' key to be released before proceeding
        keyboardAction.esc()

    if paused_flag:
        print("Game is paused, press 'T' to stop the program")
        while not keyboard.is_pressed('p'):
            time.sleep(0.1)
            if keyboard.is_pressed('t'):
                print('stopping program')
                sys.exit(0)
        paused_flag = False
        print("Game is starting")
        time.sleep(0.1)
        keyboardAction.esc()

    return paused_flag


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


# def plot_graph(episode_number, average_reward, learning_rate=lr):
#     plt.figure(figsize=(10, 5))  # Set the figure size once, not twice
#
#     plt.plot(episode_number, average_reward, label='Average Reward', color='blue')
#     plt.title('Average Reward per Episode')
#     plt.xlabel('Episode Number')
#     plt.ylabel('Average Reward')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'graph/average_rewards_per_episode_{len(episode_number)}_{learning_rate}.png')
#     plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    wait_for_sekiro_window()
    env = SekiroEnv()
    for lr in learning_rates:
        file_path = model_names[lr]
        agent = Agent(gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions)
        all_rewards = {}  # this is used to keep track of rewards for each learning rate
        for episode in range(EPISODES):
            ic.disable()

            # if detect.is_unwanted_state():
            #     if detect.is_eob():
            #         print('Boss defeated, saving model...')
            #         agent.save_model(file_path)
            #         plot_graph(episode_count, rewards)
            #         sys.exit(0)

            done = False
            total_reward = 0
            round_count = 0

            state = env.reset()  # img, agent_hp, agent_ep, boss_hp
            reshaped_state_0 = state[0].reshape(input_channels, 128, 128)

            while not done:
                paused = check_pause(False)

                action = agent.choose_action(state[0])
                next_state, reward, done, _ = env.step(action)
                reshaped_next_state_0 = next_state[0].reshape(input_channels, 128, 128)
                agent.store_data(reshaped_state_0, action, reward, reshaped_next_state_0, done)
                agent.learn()
                state = next_state
                total_reward += reward

                round_count += 1
                episode_count = episode + 1

            average_reward_per_episode = total_reward / round_count
            average_rewards.append(average_reward_per_episode)
            episode_numbers.append(episode + 1)

            if episode % save_frequency == 0:  # Save the model after every 10 episodes
                agent.save_model(file_path)

            average_reward_per_episode = total_reward / round_count
            print(f'Episode: {episode}, Average Reward: {average_reward_per_episode}')  # Print the reward for this episode
        all_rewards[lr] = average_rewards
        with open('rewards_data.json', 'w') as file:
            json.dump(all_rewards, file)
        # plot_graph(episode_numbers, average_rewards, lr)

    # read data from json file
    with open('rewards_data.json', 'r') as file:
        all_rewards = json.load(file)

    # plot
    plt.figure(figsize=(10, 5))
    for lr, rewards in all_rewards.items():
        print(f"Length of rewards: {len(rewards)}, Expected length: {EPISODES}")
        plt.plot(range(1, EPISODES + 1), rewards, label=f'LR: {lr}')

    plt.title('Average reward under different learning rates')
    plt.xlabel('Episode number')
    plt.ylabel('Average reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

