import os
import time
import tkinter as tk
import threading
from cv2 import VideoCapture, rectangle, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB
from PIL import ImageTk, Image
import pydirectinput as pd
import pyautogui
import pygetwindow as gw
import numpy as np
from datetime import timedelta

detect_time = 1
start_time = 0   # for GUI timer
start_flag = False
end_game = False
cv_enable = False

print(gw.getAllTitles())
ttt = gw.getWindowsWithTitle('Visual')[0]
while(True):
    time.sleep(1)
    print(ttt.isActive)
    break

# gta_ = gw.getWindowsWithTitle('FiveM')[0]

# offset = 10
# left = gta_.left + offset
# top = gta_.top + offset+20
# w = 800
# h = 600
# print(left, top, w, h)

img_PIL = pyautogui.screenshot(region=(100,100, 500, 500))
tk_img = ''


def run_model():
    print('in run_model')
    global start_flag, end_game, img_PIL, tk_img, imgshow, detect_time, cv_enable

    while(True):
        if end_game:
            print('ready to end game')
            break
        
        if not start_flag:
            print('wait start')
            time.sleep(0.5)
            continue

        start = time.time()

        img_PIL = pyautogui.screenshot(region=(100,100, 500, 500))
        frame = cvtColor(np.array(img_PIL), COLOR_RGB2BGR)
        
        
        fps = round(1/(time.time() - start), 1)
        

        

        if cv_enable:
            putText(frame, str(fps), (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            img_PIL = Image.fromarray(cvtColor(frame, COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(img_PIL)
            imgshow.configure(image=tk_img)
        else:
            time.sleep(detect_time/1000.0)

    
    print('end run_model')


def stop_game():
    global start_flag
    start_flag = False

def speed_low():
    global detect_time, speed_lab
    detect_time += 50
    speed_lab.configure(text='inference speed :' + str(detect_time))

def speed_fast():
    global detect_time, speed_lab
    if detect_time != 1:
        detect_time -= 50
    speed_lab.configure(text='inference speed :' + str(detect_time))

def on_closing():
    global end_game
    end_game = True
    time.sleep(0.5)
    # app.destroy()
    app.quit()
    # sys.exit()
    

def update_clock():
    global start_time, start_flag

    if start_flag:
        delta = int(time.time() - start_time)
        current_time = str(timedelta(seconds=delta))
        my_time.config(text=current_time)
    else:
        my_time.config(text='0:00:00')

    app.after(1000, update_clock) 

# class Task(threading.Thread):
#     def __init__(self, cmd):
#         threading.Thread.__init__(self)
#         self.cmd = cmd

#     def run(self):
#         global cv_enable
#         if cv_enable:
#             cv_enable = False
#             destroyAllWindows()
#         else:
#             cv_enable = True


class task:
    def newThread(self, my_str):
        if my_str == 'img_on_off':
            threading.Thread(target=self.img_on_off).start()
        elif my_str == 'start_game':
            threading.Thread(target=self.start_game).start()
    
    def img_on_off(self):
        global cv_enable
        print(cv_enable)
        if cv_enable:
            cv_enable = False
            print('close cv2')
        else:
            cv_enable = True
        print('out thread')

    def start_game(self):
        global start_flag, start_time
        if not start_flag:
            start_flag = True
            start_time = time.time()
            update_clock()


op = task()

def thread_func(my_str):
    op.newThread(my_str)



########################### main loop ###########################

t = threading.Thread(target = run_model)
t.start()

app = tk.Tk()
app.title('MY_GTA')

my_time = tk.Label(app, text = '0:00:00')
my_time.grid(column=0, row=0, ipadx=5, pady=5, sticky=tk.W+tk.N)

labelHeight = tk.Label(app, text = "Height Ratio")
labelHeight.grid(column=0, row=1, ipadx=5, pady=5, sticky=tk.W+tk.S)

start_button = tk.Button(app, text = 'Start', width=20, height=2, command=lambda: thread_func('start_game'))
start_button.grid(column=0, row=2, pady=10, sticky=tk.W)

stop_button = tk.Button(app, text = 'Stop', width=20, height=2, command=stop_game)
stop_button.grid(column=1, row=2, pady=10, sticky=tk.W)

speed_slow_btn = tk.Button(app, text = 'slow', width=20, height=2, command=speed_low)
speed_slow_btn.grid(column=0, row=3, pady=10, sticky=tk.W)

speed_lab = tk.Label(app, text = 'inference speed :' + str(detect_time))
speed_lab.grid(column=1, row=3, pady=10, sticky=tk.W)

speed_fast_btn = tk.Button(app, text = 'fast', width=20, height=2, command=speed_fast)
speed_fast_btn.grid(column=2, row=3, pady=10, sticky=tk.W)

tmp = ImageTk.PhotoImage(img_PIL)
imgshow = tk.Label(app, image=tmp)
imgshow.grid(column=0, row=4, pady=10, sticky=tk.W)

img_on_off_btn = tk.Button(app, text = 'ON/OFF', width=20, height=2, command=lambda: thread_func('img_on_off'))
img_on_off_btn.grid(column=1, row=4, pady=10, sticky=tk.W)

app.protocol("WM_DELETE_WINDOW", threading.Thread(target=on_closing).start)

app.mainloop()
print('app end')

t.join()
print('while end')

# sys.exit()



# os.system("pause")

