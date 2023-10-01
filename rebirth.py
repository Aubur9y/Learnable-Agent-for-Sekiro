import keyboardAction
import time

def rebirth():
    print("Dead, restart")
    time.sleep(6)
    keyboardAction.lock_vision()
    time.sleep(0.1)
    keyboardAction.attack()
    print("new round")

if __name__ == "__main__":
    rebirth()