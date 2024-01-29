import torch
from DQN_network import DQN_Agent
from DuellingDQN_network import DeullingDQN_Agent
from env import SekiroEnv

def run_trained_model(model_path, model_type, input_channels, height, width, n_actions):
    if model_type == 'Duelling_Dqn':
        agent = DeullingDQN_Agent(0.99, 1.0, 0.0001, input_channels, height, width, 1, n_actions)
    else:
        agent = DQN_Agent(0.99, 1.0, 0.0001, input_channels, height, width, 1, n_actions)

    agent.load_model(model_path)

    env = SekiroEnv()

    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

if __name__ == '__main__':
    model_path = 'models/Duelling_Dqn_2020-05-17_22-11-11.pth'
    model_type = 'Duelling_Dqn'
    input_channels = 3
    height = 224
    width = 224
    n_actions = 7

    total_reward = run_trained_model(model_path, model_type, input_channels, height, width, n_actions)
    print(total_reward)