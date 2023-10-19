import cv2
import numpy as np
import screen_capture
import time

def count_boss_blood(boss_blood_grayimage):
    boss_blood_count = 0
    for gray_value in boss_blood_grayimage[0]:
        if 78 > gray_value > 59:
            boss_blood_count += 1

    return boss_blood_count

    # Below is for testing gray value
    # for boss_bd_num in boss_blood_grayimage[0]:
    #     print(boss_bd_num)

def count_player_blood(player_blood_grayimage):
    player_blood_count = 0
    for gray_value in player_blood_grayimage[0]:
        if 98 > gray_value > 88:
            player_blood_count += 1

    return player_blood_count

    # Below is for testing gray value
    # for player_bd_num in player_blood_grayimage[-1]:
    #     print(player_bd_num)


if __name__ == '__main__':
    boss_blood_area = (60, 90, 212, 5)  # starting X, starting Y, width, height
    self_blood_area = (60, 560, 305, 5)
    battle_area = (360, 180, 300, 320)
    detect_area = (495, 184, 40, 20)
    # img = screen_capture.grab_screen(boss_blood)
    # img.show()
    largest_gray_value = 0
    smallest_gray_value = 255
    time.sleep(2)
    while True:
        # battle_window_gray = cv2.cvtColor(np.array(screen_capture.grab_screen(battle_area)), cv2.COLOR_RGB2GRAY)
        # self_blood_window_gray = cv2.cvtColor(np.array(screen_capture.grab_screen(self_blood_area)), cv2.COLOR_RGB2GRAY)
        # boss_blood_window_gray = cv2.cvtColor(np.array(screen_capture.grab_screen(boss_blood_area)), cv2.COLOR_RGB2GRAY)
        detect_window_gray = cv2.cvtColor(np.array(screen_capture.grab_screen(boss_blood_area)), cv2.COLOR_RGB2GRAY)
        count_100 = 0
        for pixel in detect_window_gray[0]:
            print(pixel)
            if 78 > pixel > 59:
                count_100 += 1
        print(count_100 > 0.9 * len(detect_window_gray[0]))

        # player_blood = count_player_blood(self_blood_window_gray)
        # boss_blood = count_boss_blood(boss_blood_window_gray)

        # print('Player blood: ' + str(player_blood))
        # print('Boss blood: ' + str(boss_blood))

        # cv2.imshow('Gray image player', self_blood_window_gray)
        # cv2.imshow('Gray image boss', boss_blood_window_gray)
        cv2.imshow('Gray image detect', detect_window_gray)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()




