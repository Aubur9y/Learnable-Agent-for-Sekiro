import cv2
import numpy as np
import screen_capture

def count_boss_blood(boss_blood_grayimage):
    boss_blood_count = 0
    for gray_value in boss_blood_grayimage[0]:
        if 75 > gray_value > 65:
            boss_blood_count += 1

    return boss_blood_count

    # Below is for testing gray value
    # for boss_bd_num in boss_blood_grayimage[0]:
    #     print(boss_bd_num)

def count_player_blood(player_blood_grayimage):
    player_blood_count = 0
    for gray_value in player_blood_grayimage[-1]:
        if 97 > gray_value > 92:
            player_blood_count += 1

    return player_blood_count

    # Below is for testing gray value
    # for player_bd_num in player_blood_grayimage[-1]:
    #     print(player_bd_num)


blood_area = (59, 90, 212, 475)  # starting X, starting Y, width, height
# img = screen_capture.grab_screen(boss_blood)
# img.show()
while True:
    # img = screen_capture.grab_screen(boss_blood)
    # img.show()

    screen_shot = np.array(screen_capture.grab_screen(blood_area))
    grayscale_img = cv2.cvtColor(screen_shot, cv2.COLOR_RGB2GRAY)

    boss_blood = count_boss_blood(grayscale_img)
    # print('Boss blood: ' + str(boss_blood))

    player_blood = count_player_blood(grayscale_img)
    # print('Player blood: ' + str(player_blood))

    cv2.imshow('1', grayscale_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()




