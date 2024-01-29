import gym
import numpy as np
from PPO_network import Agent

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, gamma=0.99, lr=alpha, gae_lambda=0.95, policy_clip=0.2, batch_size=batch_size, N=N, n_epoch=n_epochs)
    n_games = 300

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if learn_iters % N == 0 and learn_iters != 0:
                agent.learn()
            observation = observation_
            learn_iters += 1
        avg_score += score

        print(f"episode: {i}, score: {score}")
