import win32gui
import win32con
import win32api
import win32ui
from PIL import Image

def grab_screen(region):
    X, Y, width, height = region

    # Create a desktop window handle
    hdesktop = win32gui.GetDesktopWindow()

    # Create a device context
    hwndDC = win32gui.GetWindowDC(hdesktop)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # Create a bitmap and copy the screenshot into it
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)
    saveDC.BitBlt((0, 0), (width, height), mfcDC, (X, Y), win32con.SRCCOPY)

    # Get the bitmap as a PIL image
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )

    # Clean up
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hdesktop, hwndDC)

    return img

# Testing
# window = (0, 38, 1024, 576)  # starting X, starting Y, width, height
# resulting_img = grab_screen(window)
#
# if resulting_img:
#     resulting_img.show()