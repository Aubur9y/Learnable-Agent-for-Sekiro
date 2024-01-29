import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_channels, lr, height, width, ckpt_dir='ckpt/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(ckpt_dir, 'actor_torch_ppo')

        kernel_size = (5, 5)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout(p=0.2)

        conv_output_size = self._calculate_conv_output_size(height, width)

        self.fc = nn.Linear(conv_output_size, 256)

        self.actor = nn.Sequential(
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

        # self.actor = nn.Sequential(
        #     nn.Linear(*input_dims, fc1_dim),
        #     nn.ReLU(),
        #     nn.Linear(fc1_dim, fc2_dim),
        #     nn.ReLU(),
        #     nn.Linear(fc2_dim, n_actions),
        #     nn.Softmax(dim=-1)
        # )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = nn.ReLU()(self.conv1(state))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc(x))
        dist = self.actor(x)
        dist = Categorical(dist)
        return dist

    def _calculate_conv_output_size(self, height, width):
        # This is a helper function to calculate the output size of the conv layers
        # Assumes square kernel size and stride 1 for simplicity
        output_height = height
        output_width = width
        output_height = output_height - 2  # Two conv layers with padding=1 maintain the size
        output_width = output_width - 2
        return output_height * output_width * 64  # 64 is the number of filters in the last conv layer

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_channels, lr, height, width, ckpt_dir='ckpt/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(ckpt_dir, 'critic_torch_ppo')

        kernel_size = (5, 5)

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout(p=0.2)

        conv_output_size = self._calculate_conv_output_size(height, width)

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = nn.ReLU()(self.conv1(state))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        value = self.fc2(x)
        return value

    def _calculate_conv_output_size(self, height, width):
        # This is a helper function to calculate the output size of the conv layers
        # Assumes square kernel size and stride 1 for simplicity
        output_height = height
        output_width = width
        output_height = output_height - 2  # Two conv layers with padding=1 maintain the size
        output_width = output_width - 2
        return output_height * output_width * 64  # 64 is the number of filters in the last conv layer

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2024, n_epoch=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epoch = n_epoch
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, lr)
        self.critic = CriticNetwork(input_dims, lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_model(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epoch):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batch()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()