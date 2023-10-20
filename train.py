import sys

import cv2
import numpy as np
import torch
import logging
import icecream

import keyboardAction
from env import SekiroEnv
import detect

import time
import keyboard
import pygetwindow as gw
import matplotlib.pyplot as plt
from DQN_network import Agent

# Constants
width = 128
height = 128
input_channels = 1
battle_area = (360, 180, 300, 320)
boss_blood_area = (59, 90, 212, 475)
player_blood_area = (53, 560, 305, 5)
EPISODES = 300
episode_count = 0
input_dims = (input_channels, height, width)
n_actions = 5
batch_size = 16
gamma = 0.99
lr = 0.003
epsilon = 1.0
file_path = 'model/test_model_6.pth'
save_frequency = 5
paused = True

# def count_boss_blood(boss_blood_grayimage):
#     return sum(75 > gray_value > 65 for gray_value in boss_blood_grayimage[0])
#
#
# def count_player_blood(player_blood_grayimage):
#     return sum(98 > gray_value > 81 for gray_value in player_blood_grayimage[0])

# def take_action(action_choice):
#     if action_choice == 0:  # no action
#         pass
#     elif action_choice == 1:  # j
#         keyboardAction.attack()
#     elif action_choice == 2:  # space
#         keyboardAction.jump()
#     elif action_choice == 3:  # k
#         keyboardAction.defense()
#     elif action_choice == 4:  # shift
#         keyboardAction.dodge()


# def reward_mechanism(boss_hp, next_boss_hp, player_hp, next_player_hp, stop_flag,
#                      episode_start_time):
#     """return reward, done, stop_flag"""
#     if next_player_hp < 1:
#         return -10, 1, 0
#     elif next_boss_hp == 1:
#         return 10, 0, 0
#     elif next_boss_hp < 1:
#         return 10, 1, 0
#
#     player_blood_reward = -2 if next_player_hp < player_hp else 0
#     boss_blood_reward = 2 if next_boss_hp < boss_hp else 0
#
#     # Calculate the time difference in seconds
#     episode_duration = time.time() - episode_start_time
#
#     # Add reward for living longer
#     time_reward = max(0, episode_duration - 10) * 0.1  # Example: Reward for living up to 10 seconds
#
#     return player_blood_reward + boss_blood_reward + time_reward, 0, int(next_player_hp < player_hp)


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


def plot_graph(episode_numbers, average_rewards):
    plt.figure(figsize=(10, 5))  # This will set the figure size. You can adjust if needed.

    # x is the episode number, y is the total reward
    plt.figure(figsize=(10, 5))
    plt.plot(episode_numbers, average_rewards, label='Average Reward', color='blue')
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    wait_for_sekiro_window()
    agent = Agent(gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions)
    env = SekiroEnv()
    # check_pause(paused)
    # while detect.is_pause():
    #     paused = check_pause(True)

    rewards = []  # this is used to plot graph in the end
    # keyboardAction.lock_vision()

    for episode in range(EPISODES):
        paused = check_pause(False)

        if detect.is_unwanted_state():
            # while detect.is_pause():
            #     print(f'Episode: {episode} stopped, menu detected, press p to continue')
            #     paused = check_pause(True)
            #     while paused:
            #         paused = check_pause(True)

            if detect.is_eob():
                print('Boss defeated, saving model...')
                agent.save_model(file_path)
                plot_graph(episode_count, rewards)
                sys.exit(0)

        # self_hp_window_gray, boss_hp_window_gray, battle_window_gray = status.get_status(player_blood_area,
        #                                                                                  boss_blood_area, battle_area)
        #
        # state = cv2.resize(battle_window_gray, (width, height))
        # state = np.expand_dims(state, axis=0)  # Shape: (1, height, width)
        #
        # boss_blood = count_boss_blood(boss_hp_window_gray)
        # player_blood = count_player_blood(self_hp_window_gray)

        done = False
        total_reward = 0
        # stop = 0
        round_count = 0

        # last_time = time.time()
        state = env.reset()  # img, agent_hp, agent_ep, boss_hp
        reshaped_state_0 = state[0].reshape(1, 128, 128)

        while not done:
            paused = check_pause(False)

            # state = np.array(state).reshape(-1, input_dims)
            # print('took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()

            action = agent.choose_action(state[0])
            next_state, reward, done, _ = env.step(action)
            reshaped_next_state_0 = next_state[0].reshape(1, 128, 128)
            agent.store_data(reshaped_state_0, action, reward, reshaped_next_state_0, done)
            agent.learn()
            state = next_state
            total_reward += reward
            # take_action(action)

            # self_hp_window_gray, boss_hp_window_gray, battle_window_gray = status.get_status(player_blood_area,
            #                                                                                  boss_blood_area,
            #                                                                                  battle_area)
            #
            # next_state = cv2.resize(battle_window_gray, (width, height))
            # next_state = np.expand_dims(next_state, axis=0)  # Shape: (1, height, width)
            #
            # next_boss_blood = count_boss_blood(boss_hp_window_gray)
            # next_player_blood = count_player_blood(self_hp_window_gray)
            #
            # reward, done, stop = reward_mechanism(boss_blood, next_boss_blood, player_blood,
            #                                       next_player_blood, stop, last_time)

            # update to the next iteration
            # player_blood = next_player_blood
            # boss_blood = next_boss_blood

            rewards.append(total_reward)
            round_count += 1

            episode_count = episode + 1

        if episode % save_frequency == 0:  # Save the model after every 10 episodes
            agent.save_model(file_path)

        average_reward_per_episode = total_reward / round_count
        print(f'Episode: {episode}, Average Reward: {average_reward_per_episode}')  # Print the reward for this episode

        # if boss_blood == 1:  # boss is dead but game is not over yet
        #     print('boss dead')
        #     while not detect.is_boss_recovered():
        #         time.sleep(0.1)
        #
        # rebirth()

    # plot_graph(episode_count, rewards)
    # cv2.destroyAllWindows()

