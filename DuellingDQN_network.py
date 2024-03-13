import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import logging

class DuellingDQN(nn.Module):
    def __init__(self, lr, input_channels, n_actions, height, width):
        super(DuellingDQN, self).__init__()

        kernel_size = (5, 5)

        # same as dqn network
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            self.flat_size = self._get_conv_output((input_channels, height, width))

        # same structure as the dqn network but output is 1 instead of n_actions
        self.value_stream = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # output n_actions
        self.advantage_stream = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.HuberLoss()
        """
        loss = 1/2 * a**2 if |a| < threshold 
        or threshold * (|a| - 1/2 * threshold) otherwise
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _get_conv_output(self, shape):
        input = torch.zeros(1, *shape)
        output = self.cnn_layer(input)
        return int(np.prod(output.size()))

    def forward(self, state):
        x = self.cnn_layer(state)

        # define the value stream
        v = self.value_stream(x)

        # define the advantage stream
        a = self.advantage_stream(x)

        # combine stream
        q_values = v + (a - a.mean(dim=1, keepdim=True))
        return q_values

class DeullingDQN_Agent:
    def __init__(self, gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions,
                 max_mem_size=15000, eps_end=0.01, eps_dec=1e-4, target_update=200):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.target_update = target_update  # how often to update the target network
        self.learn_step_counter = 0  # count how many steps we have taken so far

        self.Q_eval = DuellingDQN(lr, input_channels, n_actions, height, width)
        self.Q_eval.to(self.Q_eval.device)

        self.Q_target = DuellingDQN(lr, input_channels, n_actions, height, width)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.to(self.Q_target.device)
        self.Q_target.eval()

        self.state_memory = np.zeros((self.mem_size, input_channels, height, width), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, input_channels, height, width), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_data(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size  # find the position of the first unoccupied memory
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, state):  # choose an action based on current state(observation)
        if np.random.random() > self.epsilon:
            # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(
            #     self.Q_eval.device)  # Convert numpy array to tensor, set type to float32, add batch dimension,
            state = torch.tensor(state, dtype=torch.float32).to(self.Q_eval.device)
            # omg this problem took me 2 hours to debug
            if len(state.shape) == 2:  # if the state shape is [128, 128]
                state = state.unsqueeze(0)  # make it [1, 128, 128]
            state = state.unsqueeze(0)  # add batch dimension to make it [1, 1, 128, 128]
            # and send to device
            actions = self.Q_eval.forward(state)

            q_val = torch.max(actions)
            # greedy approach
            action = torch.argmax(actions).item()

            logging.info(f"Selected action: {action}")
        else:
            q_val = 0
            action = np.random.choice(self.action_space)
            logging.info(f"Random action: {action}")

        return action, q_val

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)  # choose a subset of all memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(next_state_batch)

        q_next_values = torch.zeros_like(q_next.max(1)[0])
        q_next_values[~terminal_batch] = q_next[~terminal_batch].max(1)[0].detach()

        self.learn_step_counter += 1

        # Here I compute the target Q-values which is used to compute the loss
        q_target = reward_batch + self.gamma * q_next_values.detach()

        # I use the loss value to update the weights and biases of the evaluation network
        loss = self.Q_eval.loss(q_eval, q_target)

        logging.info(f"loss: {loss}")

        loss.backward()

        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

        # Optimizer has done the job of updating the weights and biases, or I just need to update manually(not recommended)
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

        logging.info(f"epsilon: {self.epsilon}")

        return loss.item()

    def update_target_network(self):
        """ Update the target network with the weights and biases from the evaluation network"""
        self.Q_target.load_state_dict(copy.deepcopy(self.Q_eval.state_dict()))

    def save_model(self, file_path):
        torch.save((self.Q_eval.state_dict(), self.Q_eval.optimizer.state_dict()), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        model_state_dict, optimizer_state_dict = torch.load(file_path)
        self.Q_eval.load_state_dict(model_state_dict)
        self.Q_eval.optimizer.load_state_dict(optimizer_state_dict)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        print(f"Model loaded from {file_path}")