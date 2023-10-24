import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQNnetwork(nn.Module):
    # def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
    #     super(DQNnetwork, self).__init__()
    #     self.input_dims = input_dims
    #     self.fc1_dims = fc1_dims
    #     self.fc2_dims = fc2_dims
    #     self.n_actions = n_actions
    #     self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
    #     self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    #     self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
    #     self.optimizer = optim.Adam(self.parameters(), lr=lr)
    #     self.loss = nn.HuberLoss()
    #     self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     self.to(self.device)

    def __init__(self, lr, input_channels, n_actions, height, width):
        super(DQNnetwork, self).__init__()

        # define the convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc1_input_dims = self.get_conv_output_dims((height, width))
        print(f"Flattened Dimension: {self.fc1_input_dims}")

        self.fc1 = nn.Linear(self.fc1_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.HuberLoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        # actions is a vector of size n_actions, each element represents the Q-value of each action
        actions = self.fc2(x)

        return actions

    def get_conv_output_dims(self, input_dims):
        """ This method is used to calculate the output dimension of the convolutional layers,
        instead of calculateing the value using formula, I simulate the operations of convolutional layers"""
        with torch.no_grad():
            x = torch.zeros(1, self.conv1.in_channels, *input_dims)
            x = self.conv1(x)  # simulate the operations of convolutional layer
            x = self.conv2(x)
            x = self.conv3(x)
            return int(np.prod(x.size()))  # this gives the total number of elements in the output tensor,
            # channels * width * height
            # the [1:] is to exclude the batch dimension


class Agent:
    def __init__(self, gamma, epsilon, lr, input_channels, height, width, batch_size, n_actions,
                 max_mem_size=50000, eps_end=0.01, eps_dec=5e-4, target_update=1000):
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

        self.Q_eval = DQNnetwork(lr, input_channels, n_actions, height, width)
        self.Q_eval.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        self.Q_target = DQNnetwork(lr, input_channels, n_actions, height, width)
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

            # greedy approach
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        """ Update the weights and biases of the evaluation network"""
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()  # start off with zero gradient

        max_mem = min(self.mem_cntr, self.mem_size)  # choose a subset of all memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # Computer current Q-values according to the evaluation network
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        # Compute the Q-values of the next states according to the target network,
        # the reason I use the target network is to avoid the moving target problem
        print("next_state_batch shape:", next_state_batch.shape)
        q_next = self.Q_target.forward(next_state_batch)

        # Use a mask to compute the Q-values of non-terminal states
        # and avoid modifying q_next in-place
        # This step computes the largest Q-value of the next state
        print(q_next.shape)
        q_next_values = torch.zeros_like(q_next.max(1)[0])
        q_next_values[~terminal_batch] = q_next[~terminal_batch].max(1)[0]

        self.learn_step_counter += 1

        # Update the target network once a while, this is to avoid the moving target problem
        if self.learn_step_counter % self.target_update == 0:
            self.update_target_network()

        # Here I compute the target Q-values which is used to compute the loss
        q_target = reward_batch + self.gamma * q_next_values

        # I use the loss value to update the weights and biases of the evaluation network
        loss = self.Q_eval.loss(q_eval, q_target)
        loss.backward()
        # Optimizer has done the job of updating the weights and biases, or I just need to update manually(not recommended)
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.eps_min, self.epsilon - self.eps_dec)

    def update_target_network(self):
        """ Update the target network with the weights and biases from the evaluation network"""
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def save_model(self, file_path):
        torch.save((self.Q_eval.state_dict(), self.Q_eval.optimizer.state_dict()), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        model_state_dict, optimizer_state_dict = torch.load(file_path)
        self.Q_eval.load_state_dict(model_state_dict)
        self.Q_eval.optimizer.load_state_dict(optimizer_state_dict)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        print(f"Model loaded from {file_path}")
