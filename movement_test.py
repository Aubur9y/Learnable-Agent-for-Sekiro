import keyboardAction
import time
import keyboard
import pygetwindow as gw

# time_out = time.time() + 5
# time.sleep(2)

# while True:
#     keyboardAction.move_forward()
#     time.sleep(0.5)
#     keyboardAction.move_backward()
#     time.sleep(0.5)
#     keyboardAction.move_left()
#     time.sleep(0.5)
#     keyboardAction.move_right()
#     time.sleep(0.5)
#     keyboardAction.vision_up()
#     time.sleep(0.5)
#     keyboardAction.vision_down()
#     time.sleep(0.5)
#     keyboardAction.vision_left()
#     time.sleep(0.5)
#     keyboardAction.vision_right()
#     time.sleep(0.5)
#     keyboardAction.rebirth()
#     time.sleep(0.5)

    # if time.time() > time_out:
    #     break

# Find the window by its title
window = gw.getWindowsWithTitle('Sekiro')[0]

# Set the window's position (x, y)
window.moveTo(-10, 0)  # Set the window's position to (100, 100)