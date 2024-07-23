![](https://github.com/Aubur9y/Learnable-Agent-for-Sekiro/gif_demo.gif)

Before running the code:
Ensure you have downloaded the game by either purchasing the game directly from steam or download free versions from the internet.

Make sure the game is in version 1.06 for the code to run successfully.

Download and install this mod at: https://www.nexusmods.com/sekiro/mods/381/ to ensure the CNN works as expected

In order to run the code, the game has to be completed once, or you can download a full completed save file online.

To use saves from the internet, you have to download the application called "SimpleSekiroHelper". Tutorials can be found here: https://www.nexusmods.com/sekiro/mods/274

This code requires custom key and game settings

Game settings: 
set screen resolution to 1280x720, quality settings to medium
set brightness to 10
turn off blood and subtitles

Key bindings: 
set attck to J, Deflect/Guard to K



How to run the code:
At any of the Scupltor's Idol -> commune with Sculptor's Idol -> reflection of strength -> select Genichiro Ashina -> after successfully loading in, pause the game by pressing esc -> run the following code:

To train an agent using DQN or Dueling DQN: python3 train.py {model_path} {model_type} {episode number} 
model_path: the location you wish the model to be saved
model type: either DQN or DDQN

To train an agent using PPO or PPO with transfer learning: python3 ppo_train.py {actor_ckpt} {critic_ckpt} {number of episodes} {True/False}
actor ckpt: the location you wish to store the action network's checkpoint file
critic ckpt: the location you wish to store the critic network's checkpoint file
True: enable transfer learning (ResNet)
False: CNN

To run/test a trained model in DQN or Dueling DQN: python3 run_model_dqn.py {model_path} --model_type {DQN or Duelling_DQN} --episodes {number of episodes}
model path: the path of trained model (model type must match)

To run/test a trained model in PPO: python3 run_model_ppo.py {input_actor_network_path} {input_critic_network_path} --model_type {PPO or PPO_tl} --episodes {number of episodes}


After running the code, press T to start

Models and checkpoint used in this project: https://drive.google.com/drive/folders/10P5tMmMnpKo2j45RkcCOVi6o6HJn7ExT?usp=sharing
