"""This module is credited to
URL: https://github.com/CWHer/RL-Sekiro
Author: CWHer
Date: 10/20/2023"""

GAME_NAME = "Sekiro"
PRESS_RELEASE_DELAY = 0.02

# NOTE: actions that are too frequent are ignored,
# STEP_DELAY = {
#     "attack": 0.7,
#     "defense": 0.25,
#     "jump": 0.6,
#     "forward_dodge": 0.5,
#     "backward_dodge": 0.5,
#     "leftward_dodge": 0.5,
#     "rightward_dodge": 0.5,
# }

# This setting is for game to run at 3x speed, change if necessary to fit normal game speed
STEP_DELAY = {
    "attack": 0.15,
    "defense": 0.08,
    "jump": 0.12,
    "forward_dodge": 0.1,
    "backward_dodge": 0.1,
    "leftward_dodge": 0.1,
    "rightward_dodge": 0.1,
}

# ACTION_DELAY = {
#     "attack": 0.6,
#     "defense": 0.4,
#     "jump": 0.8,
#     "forward_dodge": 0.6,
#     "backward_dodge": 0.6,
#     "leftward_dodge": 0.6,
#     "rightward_dodge": 0.6,
# }

ACTION_DELAY = {
    "attack": 0.12,
    "defense": 0.1,
    "jump": 0.2,
    "forward_dodge": 0.12,
    "backward_dodge": 0.12,
    "leftward_dodge": 0.12,
    "rightward_dodge": 0.12,
}

# AGENT_DEAD_DELAY = 10
# ROTATION_DELAY = 1
# REVIVE_DELAY = 2.2
# PAUSE_DELAY = 0.8

AGENT_DEAD_DELAY = 2.5
ROTATION_DELAY = 0.5
REVIVE_DELAY = 1.2
PAUSE_DELAY = 0.8

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

MIN_CODE_LEN = 6
MIN_HELPER_LEN = 13

MAX_AGENT_HP = 800
MAX_AGENT_EP = 300

MAX_BOSS_HP = 9887
MAX_BOSS_EP = 4083

MAP_CENTER = (-110.252, 54.077, 239.538)

SCREEN_SIZE = (720, 1280)
SCREEN_ANCHOR = (1, -721, -1, -1)  # left, top, right, bottom

FOCUS_ANCHOR = (342, 58, 942, 658)  # (342, 58, 942, 658) (570, 200, 700, 400)
FOCUS_SIZE = (224, 224)