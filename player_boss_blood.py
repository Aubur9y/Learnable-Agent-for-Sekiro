import cv2
import numpy as np
import screen_capture

# def count_boss_blood():


boss_blood = (59, 90, 230, 475)  # starting X, starting Y, width, height
# img = screen_capture.grab_screen(boss_blood)
# img.show()
while True:
    screen_shot = np.array(screen_capture.grab_screen(boss_blood))
    cv2.imshow('1', screen_shot)
    # grayscale_img = cv2.cvtColor(screen_shot, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()




