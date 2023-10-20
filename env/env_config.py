"""This module is credited to
URL: https://github.com/CWHer/RL-Sekiro
Author: CWHer
Date: 10/20/2023"""

GAME_NAME = "Sekiro"

# ------> action
# HACK: change accordingly
PRESS_RELEASE_DELAY = 0.02

# NOTE: it takes time for action to take effect,
#   STEP_DELAY ensures that the action matches its reward
STEP_DELAY = {
    "attack": 0.7,
    "defense": 0.25,
    "jump": 0.6,
    "forward_dodge": 0.5,
    "backward_dodge": 0.5,
    "leftward_dodge": 0.5,
    "rightward_dodge": 0.5,
}
# NOTE: actions that are too frequent are ignored,
#   ACTION DELAY represents the time between successive actions
ACTION_DELAY = {
    "attack": 0.6,
    "defense": 0.4,
    "jump": 0.8,
    "forward_dodge": 0.6,
    "backward_dodge": 0.6,
    "leftward_dodge": 0.6,
    "rightward_dodge": 0.6,
}

AGENT_DEAD_DELAY = 10
ROTATION_DELAY = 1
REVIVE_DELAY = 2.2
PAUSE_DELAY = 0.8

# NOTE: directX scan codes https://www.google.com/search?q=directInputKeyboardScanCodes
AGENT_KEYMAP = {
    "attack": [0x24],
    "defense": [0x25],
    "jump": [0x39],
    "forward_dodge": [0x11, 0x2A],
    "backward_dodge": [0x1F, 0x2A],
    "leftward_dodge": [0x1E, 0x2A],
    "rightward_dodge": [0x20, 0x2A],
}

ENV_KEYMAP = {
    "pause": 0x01,
    "resume": 0x01,
}
# <------

# ------> code injection
MIN_CODE_LEN = 6
MIN_HELPER_LEN = 13
# <------

# ------> agent attributes
MAX_AGENT_HP = 800
MAX_AGENT_EP = 300

MAX_BOSS_HP = 9887
MAX_BOSS_EP = 4083

MAP_CENTER = (-110.252, 54.077, 239.538)
# <------

# ------> screenshot
# HACK: (left, top, right, bottom)
SCREEN_SIZE = (720, 1280)
SCREEN_ANCHOR = (1, -721, -1, -1)

FOCUS_ANCHOR = (342, 58, 942, 658)
FOCUS_SIZE = (128, 128)

# BOSS_HP_ANCHOR = (75, 62, 348, 71)
# <------
