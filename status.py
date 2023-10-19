import cv2
import numpy as np
from screen_capture import grab_screen


def get_self_hp(region):
    self_hp_gray = cv2.cvtColor(np.array(grab_screen(region)), cv2.COLOR_RGB2GRAY)
    return self_hp_gray

def get_boss_hp(region):
    boss_hp_gray = cv2.cvtColor(np.array(grab_screen(region)), cv2.COLOR_RGB2GRAY)
    return boss_hp_gray

def get_battle_area(region):
    battle_area_gray = cv2.cvtColor(np.array(grab_screen(region)), cv2.COLOR_RGB2GRAY)
    return battle_area_gray

def get_status(self_hp_region, boss_hp_region, battle_area_region):
    """Return self hp, boss hp, and battle area"""
    return get_self_hp(self_hp_region), get_boss_hp(boss_hp_region), get_battle_area(battle_area_region)

if __name__ == '__main__':
    pass
