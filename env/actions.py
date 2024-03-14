"""This module is adapted from
URL: https://github.com/CWHer/RL-Sekiro
Author: CWHer
Date: 10/20/2023
Changes I made:
1.Refactors common action logic into a _performAction method.
2.Uses a specific ValueError exception instead of a generic RuntimeError.
3.Adds better type hinting for improved type safety.
4.Uses consistent logging levels."""

import logging
import time
from .env_config import AGENT_KEYMAP, ENV_KEYMAP, PRESS_RELEASE_DELAY
from .keyboard import PressKey, ReleaseKey


class Actor:
    def __init__(self) -> None:
        self.agent_keymap = AGENT_KEYMAP
        self.env_keymap = ENV_KEYMAP

    def _performAction(self, key: str, keymap: dict, log_type: str, action_delay: float = 0):
        if key not in keymap:
            logging.critical(f"Invalid {log_type} action: {key}")
            raise ValueError(f"Invalid {log_type} action: {key}")

        key_codes = keymap[key]

        if isinstance(key_codes, int):  # If a single key_code
            key_codes = [key_codes]

        for key_code in key_codes:
            PressKey(key_code)
        time.sleep(PRESS_RELEASE_DELAY)
        for key_code in key_codes:
            ReleaseKey(key_code)

        logger = logging.info if log_type == "agent" else logging.debug
        logger(f"{log_type} action: {key}")

        time.sleep(action_delay)

    def agentAction(self, key: str, action_delay: float = 0) -> None:
        self._performAction(key, self.agent_keymap, "agent", action_delay)

    def envAction(self, key: str, action_delay: float = 0) -> None:
        self._performAction(key, self.env_keymap, "env", action_delay)


