import screen_capture
from screen_capture import grab_screen
import cv2
import numpy as np
import keyboardAction
import time
from DQN_network import Agent
from DQN_network import DQNnetwork
from rebirth import rebirth
import keyboard
import sys
import pygetwindow as gw

def count_boss_blood(boss_blood_grayimage):
    boss_blood_count = 0
    for gray_value in boss_blood_grayimage[0]:
        if 75 > gray_value > 65:
            boss_blood_count += 1

    return boss_blood_count

def count_player_blood(player_blood_grayimage):
    player_blood_count = 0
    for gray_value in player_blood_grayimage[-1]:
        if 97 > gray_value > 92:
            player_blood_count += 1

    return player_blood_count

def take_action(action):
    if action == 0:     # n_choose
        pass
    elif action == 1:   # j
        keyboardAction.attack()
    elif action == 2:   # space
        keyboardAction.jump()
    elif action == 3:   # k
        keyboardAction.defense()
    elif action == 4:   # shift
        keyboardAction.dodge()

def reward_machanism(boss_hp, next_boss_blood, player_hp, next_player_blood, stop):
    if next_player_blood < 1:  # player is dead
        reward = 10
        done = 1
        stop = 0
        return reward, done, stop

    elif next_boss_blood - boss_hp > 15:
        reward = 20
        done = 0
        stop = 0
        return reward, done, stop

    else:
        player_blood_reward = 0
        boss_blood_reward = 0
        if next_player_blood - player_hp < -7:
            if stop == 0:
                player_blood_reward = -6
                stop = 1
            else:
                stop = 0
        if next_boss_blood - boss_hp <= -3:
            boss_blood_reward = 4
        reward = player_blood_reward + boss_blood_reward
        done = 0
        return reward, done, stop

def check_exit():  # press T to exit the program
    if keyboard.is_pressed('t'):
        print("Exiting program...")
        keyboardAction.esc()
        sys.exit()

def wait_for_sekiro_window():
    while True:
        sekiro_window = gw.getWindowsWithTitle('Sekiro')
        if sekiro_window:
            print("Found Sekiro window. Starting program...")
            window = gw.getWindowsWithTitle('Sekiro')[0]
            window.moveTo(-10, 0)  # position the window
            break
        else:
            print("Sekiro window not found. Waiting...")
            time.sleep(1)


width = 60
height = 50
battle_area = (360, 180, 300, 320)
blood_area = (59, 90, 212, 475)

EPISODES = 100

input_dims = width * height
n_actions = 5  # attack, defense, dodge, jump and no action
batch_size = 16
gamma = 0.99
lr = 0.003
epsilon = 1.0

if __name__ == '__main__':
    wait_for_sekiro_window()
    agent = Agent(gamma, epsilon, lr, input_dims, batch_size, n_actions)

    for episode in range(EPISODES):
        battle_window_gray = cv2.cvtColor(np.array(grab_screen(battle_area)), cv2.COLOR_RGB2GRAY)
        blood_window_gray = cv2.cvtColor(np.array(grab_screen(blood_area)), cv2.COLOR_RGB2GRAY)

        state = cv2.resize(battle_window_gray, (width, height))
        boss_blood = count_boss_blood(blood_window_gray)
        player_blood = count_player_blood(blood_window_gray)

        done = 0
        total_reward = 0
        stop = 0
        last_time = time.time()
        while True:
            check_exit()
            state = np.array(state).reshape(-1, input_dims)
            print('took {} seconds'.format(time.time() - last_time))
            last_time = time.time()

            action = agent.choose_action(state)
            take_action(action)

            battle_window_gray = cv2.cvtColor(np.array(grab_screen(battle_area)), cv2.COLOR_RGB2GRAY)
            blood_window_gray = cv2.cvtColor(np.array(grab_screen(blood_area)), cv2.COLOR_RGB2GRAY)

            next_state = np.array(cv2.resize(battle_window_gray, (width, height))).reshape(-1, input_dims)
            next_boss_blood = count_boss_blood(blood_window_gray)
            next_player_blood = count_player_blood(blood_window_gray)

            reward, done, stop = reward_machanism(boss_blood, next_boss_blood, player_blood,
                                                  next_player_blood, stop)

            agent.store_data(state, action, reward, next_state, done)
            agent.learn()

            # update to the next iteration
            state = next_state
            player_blood = next_player_blood
            boss_blood = next_boss_blood
            total_reward += reward

            if done == 1:
                break

        print('episode: ', episode)
        rebirth()



