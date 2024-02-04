import ctypes
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import numpy.typing as npt
from icecream import ic
from PIL import Image, ImageGrab
from utils import timeLog

from .env_config import FOCUS_ANCHOR, FOCUS_SIZE, SCREEN_ANCHOR, SCREEN_SIZE
from .memory import Memory


class Observer():
    """[summary]
    yield raw observation
    """

    def __init__(self, handle: int, memory: Memory) -> None:
        self.handle = handle
        self.memory = memory

        anchor = ctypes.wintypes.RECT()
        ctypes.windll.user32.SetProcessDPIAware(2)
        DMWA_EXTENDED_FRAME_BOUNDS = 9
        ctypes.windll.dwmapi.DwmGetWindowAttribute(
            ctypes.wintypes.HWND(self.handle),
            ctypes.wintypes.DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(anchor), ctypes.sizeof(anchor))
        self.anchor = (anchor.left, anchor.top, anchor.right, anchor.bottom)
        logging.debug(anchor)

        self.timestamp: str = ""

    def __select(self, arr: npt.NDArray, anchor: Tuple) -> npt.NDArray:
        # NOTE: C x H x W
        left, top, right, bottom = anchor
        return arr[:, top:bottom, left:right]

    # @timeLog
    def shotScreen(self) -> npt.NDArray[np.int16]:
        screen_shot = ImageGrab.grab(self.anchor)
        # NOTE: C x H x W, "RGB"
        screen_shot = np.array(screen_shot, dtype=np.int16).transpose(2, 0, 1)  # I want to convert the screenshot to channel first so its easier for pytorch to process
        screen_shot = self.__select(screen_shot, SCREEN_ANCHOR)

        if screen_shot.shape[-2:] != SCREEN_SIZE:
            logging.critical("incorrect screenshot")
            raise RuntimeError()

        return screen_shot

    @timeLog
    def state(self, screen_shot: npt.NDArray[np.int16]) -> \
            Tuple[npt.NDArray[np.uint8], float, float, float]:
        """[summary]

        State:
            image           npt.NDArray[np.uint8]
            agent_hp        float, [0, 1]
            agent_ep        float, [0, 1]
            boss_hp         float, [0, 1]
        """
        agent_hp, agent_ep, boss_hp = self.memory.getStatus()

        focus_area = Image.fromarray(self.__select(
            screen_shot, FOCUS_ANCHOR).transpose(1, 2, 0).astype(np.uint8)).convert("RGB")
        focus_area = np.array(
            focus_area.resize(FOCUS_SIZE), dtype=np.uint8)

        return focus_area, agent_hp, agent_ep, boss_hp
