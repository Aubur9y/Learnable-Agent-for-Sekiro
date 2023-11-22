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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk

plt.style.use('seaborn-v0_8-darkgrid')

parser = argparse.ArgumentParser(
    usage='''python3 train.py model/{model name}.pth'''
)
parser.add_argument('model', type=str, help='model path')

# parse the arguments
args = parser.parse_args()

if os.path.isfile(args.model):
    print(f"The model '{args.model}' already exists.")
    sys.exit(1)

# Constants
width = 128
height = 128
input_channels = 1
battle_area = (360, 180, 300, 320)
boss_blood_area = (59, 90, 212, 475)
player_blood_area = (53, 560, 305, 5)
EPISODES = 10
episode_count = 0
round_count_for_graph = 0
input_dims = (input_channels, height, width)
n_actions = 7
batch_size = 16
gamma = 0.99
lr = 0.003
epsilon = 1.0
save_frequency = 5
paused = True
episode_numbers = []
average_rewards = []
file_path = args.model

class RealTimeGraph:
    def __init__(self, title, x_label, y_label, color, position, log_scale=False):
        self.root = Tk.Tk()
        self.root.geometry(position)  # Positions the window

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        if log_scale:
            self.ax.set_yscale('log')

        self.x_data, self.y_data = [], []
        self.line, = self.ax.plot(self.x_data, self.y_data, color=color, linewidth=2, linestyle='-')
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(True)

        # self.canvas.draw()
        # plt.ion()
        # self.root.update()

    def update(self, new_x, new_y):
        self.x_data.append(new_x)
        if torch.is_tensor(new_y):  # Check if new_y is a tensor
            new_y = new_y.detach().numpy()  # Detach and convert to numpy
        self.y_data.append(new_y)
        self.line.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()
        self.root.update()

# Instantiate two graph objects
reward_graph = RealTimeGraph('Real-time Reward Tracking', 'Action Number', 'Reward', 'blue', '+0+800')
loss_graph = RealTimeGraph('Real-time Loss Tracking', 'Training Step', 'Loss', 'red', '+700+800', log_scale=True)


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


def plot_graph(episode_number, average_reward, learning_rate=lr):
    plt.figure(figsize=(10, 5))  # Set the figure size once, not twice

    plt.plot(episode_number, average_reward, label='Average Reward', color='blue')
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'graph/average_rewards_per_episode_{len(episode_number)}_{learning_rate}.png')
    plt.show()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    logging.basicConfig(level=logging.INFO)
    wait_for_sekiro_window()
    agent = Agent(gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions)
    env = SekiroEnv()
    rewards = []  # this is used to plot graph in the end
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
            loss = agent.learn()
            state = next_state
            total_reward += reward

            round_count += 1
            episode_count = episode + 1
            round_count_for_graph += 1

            reward_graph.update(round_count_for_graph, reward)
            if loss is not None:
                loss_graph.update(round_count_for_graph, loss)

        average_reward_per_episode = total_reward / round_count
        average_rewards.append(average_reward_per_episode)
        episode_numbers.append(episode + 1)

        if episode % save_frequency == 0:  # Save the model after every 10 episodes
            agent.save_model(file_path)

        average_reward_per_episode = total_reward / round_count
        print(f'Episode: {episode}, Average Reward: {average_reward_per_episode}')  # Print the reward for this episode
    plot_graph(episode_numbers, average_rewards, lr)
