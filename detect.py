import cv2
from screen_capture import grab_screen
import numpy as np

def is_eob():  # this is the region for detection such as: menu, end of battle, etc.
    detection_area_gray = cv2.cvtColor(np.array(grab_screen((495, 184, 40, 20))), cv2.COLOR_RGB2GRAY)

    # if 90 percent of the pixels are white, then it is the end of battle
    count_255 = 0
    for pixel in detection_area_gray[0]:
        if pixel == 255:
            count_255 += 1

    return count_255 > 0.9 * len(detection_area_gray[0])

def is_pause():
    dection_area_gray = cv2.cvtColor(np.array(grab_screen((60, 560, 305, 5))), cv2.COLOR_RGB2GRAY)

    # if proportion of pixels with gray value between 88 and 98 is less than 90%, then it is the menu
    count_blood_pixel = 0
    for pixel in dection_area_gray[0]:
        if 98 > pixel > 81:
            count_blood_pixel += 1

    return count_blood_pixel == 0

# def not_in_boss_fight():
#     dection_area_gray = cv2.cvtColor(np.array(grab_screen((60, 90, 212, 5))), cv2.COLOR_RGB2GRAY)
#
#     # if proportion of pixels with gray value between 59 and 78 is less than 90%, then it is not in boss fight
#     count_blood_pixel = 0
#     for pixel in dection_area_gray[0]:
#         if 78 > pixel > 59:
#             count_blood_pixel += 1
#
#     return count_blood_pixel < 0.9 * len(dection_area_gray[0])

def is_unwanted_state():
    return is_eob()

# def is_emergence():  # this can only happen when agent choose to die rather than rebirth
#     detection_area_gray = cv2.cvtColor(np.array(grab_screen((60, 560, 305, 5))), cv2.COLOR_RGB2GRAY)
#
#     # if 90 percent of the pixels are white, then it is the end of battle
#     count_0 = 0
#     for pixel in detection_area_gray[0]:
#         if pixel == 0:
#             count_0 += 1
#
#     return count_0 > 0.95 * len(detection_area_gray[0])

def is_boss_recovered():
    dection_area_gray = cv2.cvtColor(np.array(grab_screen((60, 90, 212, 5))), cv2.COLOR_RGB2GRAY)

    # if proportion of pixels with gray value between 59 and 78 is greater than 90%, then boss has recovered
    count_blood_pixel = 0
    for pixel in dection_area_gray[0]:
        if 78 > pixel > 59:
            count_blood_pixel += 1

    return count_blood_pixel > 0.9 * len(dection_area_gray[0])

if __name__ == '__main__':
    while True:
        print(is_pause())