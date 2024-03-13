"""This module is adapted from
URL: https://github.com/CWHer/RL-Sekiro
Author: CWHer
Date: 10/20/2023
Changes I made:
1.Broke down some methods into smaller parts for better readability.
2.Improved docstrings for clarity.
3.Moved some redundant logic into separate methods for modularity."""


import logging
import time
from typing import List, Tuple, Optional

import numpy as np
import numpy.typing as npt
import win32con
import win32gui
from utils import timeLog

from .actions import Actor
from .env_config import (AGENT_DEAD_DELAY, AGENT_KEYMAP, GAME_NAME, MAP_CENTER,
                         PAUSE_DELAY, ROTATION_DELAY, STEP_DELAY)
from .memory import Memory
from .observation import Observer


class SekiroEnv:
    boss_death_count = 0
    player_death_count = 0
    steps_without_being_hit = 0

    def __init__(self) -> None:
        self.handle = self.__get_game_handle()
        self.actor = Actor()
        self.memory = Memory()
        self.observer = Observer(self.handle, self.memory)

        self.last_agent_hp = 0
        self.last_agent_ep = 0
        self.last_boss_hp = 0

    def __get_game_handle(self) -> int:
        handle = win32gui.FindWindow(0, GAME_NAME)
        if handle == 0:
            logging.critical(f"can't find {GAME_NAME}")
            raise RuntimeError()
        return handle

    def actionSpace(self) -> List[int]:
        return list(range(len(AGENT_KEYMAP)))

    def __calculate_reward(self, agent_hp: float, agent_ep: float, boss_hp: float) -> float:
        # time_penalty = -1
        # boss_max_hp = 1.00
        # boss_hp_percent = boss_hp / boss_max_hp
        # reward_multiplier = min(5.0, 1 / boss_hp_percent if boss_hp_percent != 0 else 1)
        # damage_dealt = self.last_boss_hp - boss_hp
        # boss_reward = damage_dealt * reward_multiplier
        #
        # rewards = np.array(
        #     [agent_hp - self.last_agent_hp,
        #      boss_reward,
        #      min(0.0, self.last_agent_ep - agent_ep)]
        # )
        # weights = np.array([200, 100, 50])
        # reward = weights.dot(rewards).item()
        # reward = -50 * self.last_agent_hp if agent_hp == 0 else reward
        #
        # self.last_agent_hp = agent_hp
        # self.last_agent_ep = agent_ep
        # self.last_boss_hp = boss_hp

        # if agent_hp == 0:
        #     reward = -40
        # elif boss_hp == 0:
        #     reward = 160
        # else:
        #     self_hp_reward = 0
        #     boss_hp_reward = 0
        #     if agent_hp < self.last_agent_hp:  # agent got hit
        #         self_hp_reward = -20
        #     if self.last_boss_hp - boss_hp > 0.05:  # boss got hit, and it's not caused by jumping over the boss
        #         boss_hp_reward = 80
        #     reward = self_hp_reward + boss_hp_reward
        #
        # self.last_agent_hp = agent_hp
        # self.last_agent_ep = agent_ep
        # self.last_boss_hp = boss_hp

        reward = 0

        if agent_hp == 0:
            return 10

        if self.last_boss_hp - boss_hp < 0:  # this means boss has died
            return 20

        # if self.last_agent_hp == agent_hp and self.last_agent_ep == agent_ep:
        #     self.steps_without_being_hit += 1
        # else:
        #     self.steps_without_being_hit = 0

        self_hp_reduced = self.last_agent_hp - agent_hp
        self_ep_reduced = self.last_agent_ep - agent_ep
        boss_hp_reduced = self.last_boss_hp - boss_hp

        if self_hp_reduced > 0:
            reward -= self_hp_reduced * 40

        if self_ep_reduced > 0:
            reward -= self_ep_reduced * 5

        if boss_hp_reduced > 0.05:
            reward += boss_hp_reduced * 100

        # if self.steps_without_being_hit >= 20:
        #     reward += 0.5
        #     self.steps_without_being_hit = 0

        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp

        return reward

    def step(self, action: int) -> Tuple[Tuple[npt.NDArray[np.uint8], float, float, float], float, bool]:
        self.memory.resetEndurance()

        lock_state = self.memory.lockBoss()
        logging.info(f"lock state: {lock_state}")

        action_key = list(AGENT_KEYMAP.keys())[action]
        self.actor.agentAction(action_key, action_delay=STEP_DELAY[action_key])

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)

        agent_hp, agent_ep, boss_hp = state[-3:]
        boss_hp = self.__handle_boss_death(boss_hp)
        reward = self.__calculate_reward(agent_hp, agent_ep, boss_hp)

        self.__update_status(agent_hp, agent_ep, boss_hp)

        done = self.__handle_agent_death(agent_hp)

        logging.info(f"agent hp: {agent_hp:.4f}, agent ep: {agent_ep:.4f}, boss hp: {boss_hp:.4f}")
        logging.info(f"reward: {reward:<.2f}, done: {done}")

        return state, reward, done

    def __handle_boss_death(self, boss_hp: float) -> float:
        if boss_hp < 0.1:
            self.boss_death_count += 1
            self.memory.reviveBoss()
            boss_hp = 1.00

        return boss_hp

    def __update_status(self, agent_hp: float, agent_ep: float, boss_hp: float) -> None:
        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp

    def __handle_agent_death(self, agent_hp: float) -> bool:
        if agent_hp == 0:
            self.player_death_count += 1
            time.sleep(AGENT_DEAD_DELAY)
            self.memory.lockBoss()
            time.sleep(ROTATION_DELAY)
            self.memory.reviveAgent(need_delay=True)
            self.actor.envAction("pause")
            time.sleep(2)
            return True
        return False

    def reset(self) -> Tuple[npt.NDArray[np.uint8], float, float, float]:
        self.__restore_and_focus_window()

        self.memory.transportAgent(MAP_CENTER)
        self.memory.lockBoss()
        self.actor.envAction("resume", action_delay=PAUSE_DELAY)

        screen_shot = self.observer.shotScreen()
        state = self.observer.state(screen_shot)
        self.last_agent_hp, self.last_agent_ep, self.last_boss_hp = state[-3:]

        return state

    def __restore_and_focus_window(self) -> None:
        win32gui.SendMessage(self.handle, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
        win32gui.SetForegroundWindow(self.handle)
        time.sleep(0.5)

    def get_boss_death_count(self) -> int:
        return self.boss_death_count

    def get_player_death_count(self) -> int:
        return self.player_death_count
