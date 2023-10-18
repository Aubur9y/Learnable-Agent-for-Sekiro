import cv2
import numpy as np
import keyboardAction
import time
import keyboard
import pygetwindow as gw
import matplotlib.pyplot as plt
from screen_capture import grab_screen
from DQN_network import Agent
from rebirth import rebirth
import pyautogui


def count_boss_blood(boss_blood_grayimage):
    return sum(75 > gray_value > 65 for gray_value in boss_blood_grayimage[0])

def count_player_blood(player_blood_grayimage):
    return sum(97 > gray_value > 92 for gray_value in player_blood_grayimage[0])


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


def reward_mechanism(boss_hp, next_boss_hp, player_hp, next_player_hp, stop_flag,
                     episode_start_time):
    if next_player_hp < 1:
        return -10, 1, 0
    elif next_boss_hp <= 1:
        return 10, 1, 0

    player_blood_reward = -2 if next_player_hp < player_hp else 0
    boss_blood_reward = 2 if next_boss_hp < boss_hp else 0

    # Calculate the time difference in seconds
    episode_duration = time.time() - episode_start_time

    # Add reward for living longer
    time_reward = max(0, episode_duration - 10) * 0.1  # Example: Reward for living up to 10 seconds

    return player_blood_reward + boss_blood_reward + time_reward, 0, int(next_player_hp < player_hp)


def check_pause(paused_flag):
    """
    Toggles the paused flag when 'p' is pressed.
    Returns the updated paused state.
    """
    if keyboard.is_pressed('p'):
        paused_flag = not paused_flag  # Toggle the paused state
        print("Game is {}".format("paused" if paused_flag else "starting"))
        time.sleep(2)  # has to wait for current motion to end in order to pause
        # Wait for the 'p' key to be released before proceeding
        keyboardAction.esc()
        print("2")

    if paused_flag:
        print("Game is paused")
        print("1")
        while not keyboard.is_pressed('p'):
            time.sleep(0.1)
        paused_flag = False
        print("Game is starting")
        time.sleep(0.1)
        keyboardAction.esc()
        print("3")

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


def is_unwanted_state(image):
    return np.any((image[-1] == 208) | (image[-1] == 209) | (image[-1] == 0))

width = 84
height = 84
input_channels = 1  # 1 for grayscale, 3 for RGB
battle_area = (360, 180, 300, 320)
boss_blood_area = (59, 90, 212, 475)
player_blood_area = (60, 560, 305, 5)

EPISODES = 1000

input_dims = (input_channels, height, width)
n_actions = 5  # attack, defense, dodge, jump and no action
batch_size = 16
gamma = 0.99
lr = 0.003
epsilon = 1.0

file_path = 'model/test_model_5.pth'
save_frequency = 5

paused = False

if __name__ == '__main__':
    wait_for_sekiro_window()
    agent = Agent(gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions)
    # check_pause(paused)
    starting_window = cv2.cvtColor(np.array(grab_screen(player_blood_area)), cv2.COLOR_RGB2GRAY)
    if is_unwanted_state(starting_window):
        paused = check_pause(True)

    rewards = []  # this is used to plot graph in the end
    keyboardAction.lock_vision()

    for episode in range(EPISODES):
        paused = check_pause(False)

        battle_window_gray = cv2.cvtColor(np.array(grab_screen(battle_area)), cv2.COLOR_RGB2GRAY)

        # count hp for both boss and player
        boss_blood_window_gray = cv2.cvtColor(np.array(grab_screen(boss_blood_area)), cv2.COLOR_RGB2GRAY)
        player_blood_window_gray = cv2.cvtColor(np.array(grab_screen(player_blood_area)), cv2.COLOR_RGB2GRAY)

        if is_unwanted_state(player_blood_window_gray):
            print(f'Episode: {episode} stopped due to unwanted state!')
            agent.save_model(file_path)
            paused = check_pause(True)  # If the game is in an unwanted state, force pause
            if paused:
                continue  # Skip to the next episode

        state = cv2.resize(battle_window_gray, (width, height))
        state = np.expand_dims(state, axis=0)  # Shape: (1, height, width)

        boss_blood = count_boss_blood(boss_blood_window_gray)
        player_blood = count_player_blood(player_blood_window_gray)

        done = 0
        total_reward = 0
        stop = 0
        round_count = 0

        last_time = time.time()

        while True:
            paused = check_pause(False)

            # state = np.array(state).reshape(-1, input_dims)
            print('took {} seconds'.format(time.time() - last_time))
            last_time = time.time()

            action = agent.choose_action(state)
            take_action(action)

            battle_window_gray = cv2.cvtColor(np.array(grab_screen(battle_area)), cv2.COLOR_RGB2GRAY)

            # count hp for both boss and player
            boss_blood_window_gray = cv2.cvtColor(np.array(grab_screen(boss_blood_area)), cv2.COLOR_RGB2GRAY)
            player_blood_window_gray = cv2.cvtColor(np.array(grab_screen(player_blood_area)), cv2.COLOR_RGB2GRAY)

            next_state = cv2.resize(battle_window_gray, (width, height))
            next_state = np.expand_dims(next_state, axis=0)  # Shape: (1, height, width)

            next_boss_blood = count_boss_blood(boss_blood_window_gray)
            next_player_blood = count_player_blood(player_blood_window_gray)

            reward, done, stop = reward_mechanism(boss_blood, next_boss_blood, player_blood,
                                                  next_player_blood, stop, last_time)

            agent.store_data(state, action, reward, next_state, done)
            agent.learn()

            # update to the next iteration
            state = next_state
            player_blood = next_player_blood
            boss_blood = next_boss_blood

            total_reward += reward
            rewards.append(total_reward)
            round_count += 1

            if done == 1:
                break

        if episode % save_frequency == 0:  # Save the model after every 10 episodes
            agent.save_model(file_path)

        average_reward_per_episode = total_reward / round_count
        print(f'Episode: {episode}, Average Reward: {average_reward_per_episode}')  # Print the reward for this episode
        if player_blood < 1:
            rebirth()

    plt.figure(figsize=(10, 5))  # This will set the figure size. You can adjust if needed.
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Rewards over Episodes')
    plt.grid(True)  # This will add a grid for better readability
    plt.tight_layout()
    plt.savefig("rewards_vs_episodes.png")  # Optionally, save the figure to a file
    plt.show()

    cv2.destroyAllWindows()

