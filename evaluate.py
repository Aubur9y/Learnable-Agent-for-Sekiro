import torch
from DQN_network import DQNnetwork


width = 84
height = 84
input_channels = 1  # 1 for grayscale, 3 for RGB

input_dims = (input_channels, height, width)
n_actions = 5  # attack, defense, dodge, jump and no action
batch_size = 16
gamma = 0.99
# lr = 0.003
# epsilon = 1.0

learning_rates = [0.001, 0.003, 0.005]
epsilons = [0.5, 0.7, 1.0]



model_state = torch.load('model/test_model_5.pth')[0]

for lr in learning_rates:
    for epsilon in epsilons:
        print(f"Testing with Learning Rate: {lr} and Epsilon: {epsilon}")

        model = DQNnetwork(lr, input_channels, n_actions, height, width)
        model.load_state_dict(model_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # add game loop here

        model.eval()

# content = torch.load('model/test_model_5.pth')
# print(type(content))
