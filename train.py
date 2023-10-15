from screen_capture import grab_screen
import cv2
import numpy as np
import keyboardAction
import time
from DQN_network import Agent
from rebirth import rebirth
import keyboard
import pygetwindow as gw
import matplotlib.pyplot as plt


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


def take_action(action_choice):
    if action_choice == 0:  # no action
        pass
    elif action_choice == 1:  # j
        keyboardAction.attack()
    elif action_choice == 2:  # space
        keyboardAction.jump()
    elif action_choice == 3:  # k
        keyboardAction.defense()
    elif action_choice == 4:  # shift
        keyboardAction.dodge()


# def reward_machanism(boss_hp, next_boss_blood, player_hp, next_player_blood, stop, emergency_break):
#     if next_player_blood < 1:  # player is dead
#         if emergency_break < 2:
#             reward = -10
#             done = 1
#             stop = 0
#             return reward, done, stop, emergency_break
#         else:
#             reward = -10
#             done = 1
#             stop = 0
#             emergency_break = 100
#             return reward, done, stop, emergency_break
#
#     elif next_boss_blood - boss_hp > 15:
#         if emergency_break < 2:
#             reward = 20
#             done = 0
#             stop = 0
#             emergency_break += 1
#             return reward, done, stop, emergency_break
#         else:
#             reward = 20
#             done = 0
#             stop = 0
#             emergency_break = 100
#             return reward, done, stop, emergency_break
#     else:
#         player_blood_reward = 0
#         boss_blood_reward = 0
#         if next_player_blood - player_hp < -5:
#             if stop == 0:
#                 player_blood_reward = -6
#                 stop = 1
#         else:
#             stop = 0
#         if next_boss_blood - boss_hp <= -3:
#             boss_blood_reward = 4
#         reward = player_blood_reward + boss_blood_reward
#         done = 0
#         emergence_break = 0
#         return reward, done, stop, emergence_break
def reward_mechanism(boss_hp, next_boss_hp, player_hp, next_player_hp, stop_flag,
                     emergency_break_flag, episode_start_time):
    if next_player_hp < 1:
        if emergency_break_flag < 2:
            return -10, 1, 0, 0
        else:
            return -10, 1, 0, 100

    player_blood_reward = -2 if next_player_hp < player_hp else 0.1
    boss_blood_reward = 2 if next_boss_hp < boss_hp else -0.1

    # Calculate the time difference in seconds
    episode_duration = time.time() - episode_start_time

    # Add reward for living longer
    time_reward = max(0, episode_duration - 10) * 0.1  # Example: Reward for living up to 10 seconds

    return player_blood_reward + boss_blood_reward + time_reward, 0, int(next_player_hp < player_hp), 0


def check_pause(paused_flag):  # press P to exit the program
    if keyboard.is_pressed('p'):
        if paused_flag:
            paused_flag = False
            print("Game is starting")
            time.sleep(1)
            keyboardAction.esc()
        else:
            paused_flag = True
            print("Game is paused")
            time.sleep(1)
            keyboardAction.esc()
    if paused_flag:
        print("Game is paused")
        while True:
            if keyboard.is_pressed('p'):
                if paused_flag:
                    paused_flag = False
                    print("Game is starting")
                    time.sleep(1)
                    keyboardAction.esc()
                    break
                else:
                    paused_flag = True
                    time.sleep(1)
                    keyboardAction.esc()
    return paused_flag


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


width = 96
height = 88
battle_area = (360, 180, 300, 320)
blood_area = (59, 90, 212, 475)

EPISODES = 100

input_dims = width * height
n_actions = 5  # attack, defense, dodge, jump and no action
batch_size = 16
gamma = 0.99
lr = 0.003
epsilon = 1.0

file_path = 'model/test_model_3.pth'
save_frequency = 5

paused = True

if __name__ == '__main__':
    wait_for_sekiro_window()
    agent = Agent(gamma, epsilon, lr, input_dims, batch_size, n_actions)

    emergency_break = 0
    paused = check_pause(paused)

    rewards = []  # this is used to plot graph in the end
    keyboardAction.lock_vision()

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

            reward, done, stop, emergency_break = reward_mechanism(boss_blood, next_boss_blood, player_blood,
                                                                   next_player_blood, stop, emergency_break,
                                                                   last_time)

            if emergency_break == 100:
                # emergence break , save model and paused
                print("emergency break")
                agent.save_model(file_path)
                paused = True

            agent.store_data(state, action, reward, next_state, done)
            agent.learn()

            # update to the next iteration
            state = next_state
            player_blood = next_player_blood
            boss_blood = next_boss_blood
            total_reward += reward
            rewards.append(total_reward)
            paused = check_pause(paused)

            print('reward {} '.format(reward))

            if done == 1:
                break

        if episode % save_frequency == 0:  # Save the model after every 10 episodes
            agent.save_model(file_path)

        print(f'Episode: {episode}, Total Reward: {total_reward}')  # Print the reward for this episode
        rebirth()

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards over Episodes')
    plt.show()
