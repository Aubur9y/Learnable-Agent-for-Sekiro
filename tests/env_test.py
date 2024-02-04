import win32gui
from numpy import ndarray, dtype, unsignedinteger
from numpy._typing import _8Bit

from env.observation import Observer
from env.memory import Memory
import cv2
import numpy.typing as npt

import ctypes
import logging
from datetime import datetime
from typing import Tuple, Any

import numpy as np
from icecream import ic
from PIL import Image, ImageGrab
from utils import timeLog

from env.env_config import FOCUS_ANCHOR, FOCUS_SIZE, SCREEN_ANCHOR, SCREEN_SIZE
from env.memory import Memory


# handle = win32gui.FindWindow(0, "Sekiro")
# memory = Memory()
#
# observer = Observer(handle, memory)
handle = win32gui.FindWindow(0, "Sekiro")
anchor = ctypes.wintypes.RECT()
ctypes.windll.user32.SetProcessDPIAware(2)
DMWA_EXTENDED_FRAME_BOUNDS = 9
ctypes.windll.dwmapi.DwmGetWindowAttribute(
    ctypes.wintypes.HWND(handle),
    ctypes.wintypes.DWORD(DMWA_EXTENDED_FRAME_BOUNDS),
    ctypes.byref(anchor), ctypes.sizeof(anchor))
anchor = (anchor.left, anchor.top, anchor.right, anchor.bottom)


def __select(arr: npt.NDArray, anchor: Tuple) -> npt.NDArray:
    # NOTE: C x H x W
    left, top, right, bottom = anchor
    return arr[:, top:bottom, left:right]

def shotScreen() -> npt.NDArray[np.int16]:
    screen_shot = ImageGrab.grab(anchor)

    # NOTE: C x H x W, "RGB"
    screen_shot = np.array(screen_shot, dtype=np.int16).transpose(2, 0, 1)  # I want to convert the screenshot to channel first so its easier for pytorch to process
    screen_shot = __select(screen_shot, SCREEN_ANCHOR)

    print(screen_shot.shape)

    if screen_shot.shape[-2:] != SCREEN_SIZE:
        logging.critical("incorrect screenshot")
        raise RuntimeError()

    return screen_shot

def state(screen_shot: npt.NDArray[np.int16]) -> \
        ndarray[Any, dtype[Any]]:
    """[summary]

    State:
        image           npt.NDArray[np.uint8]
        agent_hp        float, [0, 1]
        agent_ep        float, [0, 1]
        boss_hp         float, [0, 1]
    """

    focus_area = Image.fromarray(__select(
        screen_shot, FOCUS_ANCHOR).transpose(1, 2, 0).astype(np.uint8)).convert("RGB")
    focus_area = np.array(
        focus_area.resize(FOCUS_SIZE), dtype=np.uint8)

    return focus_area

for i in range(1):
    screen_shot = shotScreen()

    state = state(screen_shot)

    # screen_shot = screen_shot.transpose(1, 2, 0)

    if screen_shot is not None:
        print(screen_shot.shape)  # This should print the dimensions of the image
        cv2.imshow("screen", state)
    else:
        print("Failed to load screenshot.")

    cv2.waitKey(0)



