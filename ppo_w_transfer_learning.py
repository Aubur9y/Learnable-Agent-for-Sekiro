import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import neptune
import logging
from torch.distributions.categorical import Categorical
from torchvision import models

# run = neptune.init_run(
#     project="aubury/sekiro",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2MTAyMTdhMS03YWRmLTQ4YzUtYTE5Zi0yYTU2OTQxNzVkM2QifQ==",
# )

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
        # if isinstance(state, (list, tuple)):
        #     print(f"State is a {type(state)} with length {len(state)}")
        #     for i, item in enumerate(state):
        #         print(
        #             f"Element {i}: Type={type(item)}, Shape={np.array(item).shape if isinstance(item, np.ndarray) else 'N/A'}")
        # else:
        #     print(f"State is not a list or tuple, Type={type(state)}")
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

        logging.info('Creating actor network...')

        # self.base_model = models.resnet18(pretrained=True)
        self.base_model = models.resnet34(pretrained=True)

        # freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # unfreeze the last 3 layers
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        self.checkpoint_file = os.path.join(ckpt_dir, 'actor_torch_ppo_tl')

        if input_channels != 3:
            self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                              bias=False)

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.base_model(state)

    def save_checkpoint(self, file_path):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(file_path))

class CriticNetwork(nn.Module):
    def __init__(self, input_channels, lr, height, width, ckpt_dir='ckpt/ppo'):
        super(CriticNetwork, self).__init__()

        logging.info('Creating critic network...')

        self.checkpoint_file = os.path.join(ckpt_dir, 'critic_torch_ppo_tl')

        # self.base_model = models.resnet18(pretrained=True)
        self.base_model = models.resnet34(pretrained=True)

        # freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # unfreeze the last 3 layers
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        if input_channels != 3:
            self.base_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                              bias=False)

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.base_model(state)

    def save_checkpoint(self, file_path):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(file_path))

class PPO_Agent_tl:
    def __init__(self, n_actions, input_channels, height, width, gamma=0.99, lr=0.0003, gae_lambda=0.95, policy_clip=0.2, batch_size=64, N=2024, n_epoch=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epoch = n_epoch
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_channels, lr, height, width)
        self.critic = CriticNetwork(input_channels, lr, height, width)
        self.memory = PPOMemory(batch_size)

    def store_data(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_model(self, file_path_actor, file_path_critic):
        self.actor.save_checkpoint(file_path_actor)
        self.critic.save_checkpoint(file_path_critic)

    def load_model(self, file_path_actor, file_path_critic):
        self.actor.load_checkpoint(file_path_actor)
        self.critic.load_checkpoint(file_path_critic)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.actor.device)
        state = state.unsqueeze(0)

        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        value = self.critic(state)

        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        logging.info(f"Selected action: {action}")

        # probs = dist.log_prob(action)
        #
        # val = self.critic(state)
        #
        # action = action.cpu.numpy()
        # probs = probs.cpu().numpy()
        # val = val.item()

        return action, probs, value

    def learn(self):
        total_actor_loss = 0
        total_critic_loss = 0
        total_loss = 0
        num_batches = 0

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
                num_batches += 1

                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                action_probs = self.actor(states)
                dist = Categorical(action_probs)

                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]

                entropy = dist.entropy().mean()
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - 0.01 * entropy
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                batch_total_loss = actor_loss + 0.5 * critic_loss

                # adding up all the losses
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_loss += batch_total_loss.item()

                # run["ppo/train/batch/loss"].append(total_loss)

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                batch_total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        avg_actor_loss = total_actor_loss / num_batches
        avg_critic_loss = total_critic_loss / num_batches
        avg_total_loss = total_loss / num_batches

        logging.info(f"avg_total_loss: {avg_total_loss}")

        self.memory.clear_memory()

        return avg_total_loss