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
        rewards = np.array([
            agent_hp - self.last_agent_hp,
            self.last_boss_hp - boss_hp if boss_hp <= self.last_boss_hp else self.last_boss_hp + 1 - boss_hp
        ])
        weights = np.array([10, 200])
        return weights.dot(rewards).item()

    @timeLog
    def step(self, action: int) -> Tuple[Tuple[npt.NDArray[np.uint8], float, float, float], float, bool, None]:
        """
        Perform the action and observe the result.

        Returns:
            state (Tuple): The state which consists of the screenshot, agent hp, agent ep, and boss hp.
            reward (float): The reward for performing the action.
            done (bool): Whether the agent is done with its actions.
            info (None): Placeholder for additional info, not used currently.
        """
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

        return state, reward, done, None

    def __handle_boss_death(self, boss_hp: float) -> float:
        if boss_hp < 0.1:
            self.memory.reviveBoss()
            boss_hp = 1.00
        return boss_hp

    def __update_status(self, agent_hp: float, agent_ep: float, boss_hp: float) -> None:
        self.last_agent_hp = agent_hp
        self.last_agent_ep = agent_ep
        self.last_boss_hp = boss_hp

    def __handle_agent_death(self, agent_hp: float) -> bool:
        if agent_hp == 0:
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