from time import time
import pyautogui
import time
import random
import pydirectinput as pd
import cv2
import numpy as np

def sight_r():
    pyautogui.moveRel(100, 0, duration = 0.5)
    time.sleep(0.5)

def sight_l():
    pyautogui.moveRel(-100, 0, duration = 0.5)
    time.sleep(0.5)

def sight_down():
    pyautogui.moveRel(0, 100, duration = 0.5)
    time.sleep(0.5)

def sight_up():
    pyautogui.moveRel(0, -100, duration = 0.5)
    time.sleep(0.5)


time.sleep(5)
print('start')


im = pyautogui.screenshot(region=(100,100, 1000, 1000))
img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

time.sleep(0.5)

# # pd.press('x')
# # time.sleep(random.uniform(0.1, 0.2))
pd.keyDown('w')


sight_up()
sight_down()
sight_l()
sight_r()

pd.keyUp('w')
time.sleep(0.5)

cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('DONE')
