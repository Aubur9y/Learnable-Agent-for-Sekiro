import ctypes
from ctypes import wintypes
import time

user32 = ctypes.WinDLL('user32', use_last_error=True)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MAPVK_VK_TO_VSC = 0

# wasd is control, arrow keys control the attack direction
W = 0x57
A = 0x41
S = 0x53
D = 0x44
R = 0x52

J = 0x4A
K = 0x4B
# I = 0x49
# L = 0x4C

ESC = 0x1B
SPACE = 0x20
SHIFT = 0x10
C = 0x43

V = 0x56

Arr_UP = 0x26
Arr_DOWN = 0x28
Arr_LEFT = 0x25
Arr_RIGHT = 0x27

# C struct definitions

wintypes.ULONG_PTR = wintypes.WPARAM


PUL = ctypes.POINTER(ctypes.c_ulong)


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", PUL))


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", PUL))

    def __init__(self, *args, **kwds):
        super(KEYBDINPUT, self).__init__(*args, **kwds)
        # some programs use the scan code even if KEYEVENTF_SCANCODE
        # isn't set in dwFflags, so attempt to map the correct code.
        if not self.dwFlags & KEYEVENTF_UNICODE:
            self.wScan = user32.MapVirtualKeyExW(self.wVk,
                                                 MAPVK_VK_TO_VSC, 0)


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD))


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = (("ki", KEYBDINPUT),
                    ("mi", MOUSEINPUT),
                    ("hi", HARDWAREINPUT))

    _anonymous_ = ("_input",)
    _fields_ = (("type", wintypes.DWORD),
                ("_input", _INPUT))


LPINPUT = ctypes.POINTER(INPUT)


def _check_count(result, func, args):
    if result == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


user32.SendInput.errcheck = _check_count
user32.SendInput.argtypes = (wintypes.UINT,  # nInputs
                             LPINPUT,  # pInputs
                             ctypes.c_int)  # cbSize


# This area is where the actions take place
def PressKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    x = INPUT(type=INPUT_KEYBOARD,
              ki=KEYBDINPUT(wVk=hexKeyCode,
                            dwFlags=KEYEVENTF_KEYUP))
    user32.SendInput(1, ctypes.byref(x), ctypes.sizeof(x))


def move_forward():
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)

def move_backward():
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)

def move_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)

def move_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)

def lock_vision():
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)

def attack():
    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)

def defense():
    PressKey(K)
    time.sleep(0.05)
    ReleaseKey(K)

def vision_up():
    PressKey(Arr_UP)
    time.sleep(0.01)
    ReleaseKey(Arr_UP)

def vision_down():
    PressKey(Arr_DOWN)
    time.sleep(0.01)
    ReleaseKey(Arr_DOWN)

def vision_left():
    PressKey(Arr_LEFT)
    time.sleep(0.01)
    ReleaseKey(Arr_LEFT)

def vision_right():
    PressKey(Arr_RIGHT)
    time.sleep(0.01)
    ReleaseKey(Arr_RIGHT)

def jump():
    PressKey(SPACE)
    time.sleep(0.1)
    ReleaseKey(SPACE)

def dodge():
    PressKey(SHIFT)
    time.sleep(0.1)
    ReleaseKey(SHIFT)

def esc():
    PressKey(ESC)
    time.sleep(0.1)
    ReleaseKey(ESC)

def rebirth():
    PressKey(J)
    time.sleep(0.1)
    ReleaseKey(J)

# def new_game_after_death():
#     PressKey(SPACE)
#     time.sleep(0.01)
#     ReleaseKey(SPACE)



    


