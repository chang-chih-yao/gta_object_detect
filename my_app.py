import os
import time
import tkinter as tk
from tkinter import messagebox
import threading
from cv2 import VideoCapture, rectangle, imshow, waitKey, imread, destroyAllWindows, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_RGB2BGR
from PIL import ImageTk
import pydirectinput as pd
import pyautogui
import numpy as np
from datetime import timedelta

detect_time = 401
start_flag = False
end_game = False
img_PIL = pyautogui.screenshot(region=(100,100, 500, 500))
tk_img = ''
start_time = 0

def run_model():
    print('in run_model')
    global start_flag, end_game, img_PIL, tk_img, cv2show, detect_time

    while(True):
        

        if not start_flag:
            destroyAllWindows()
            print('wait start')
            time.sleep(0.5)
            continue

        img_PIL = pyautogui.screenshot(region=(100,100, 500, 500))
        if end_game:
            print('ready to end game')
            time.sleep(0.5)
            destroyAllWindows()
            break
        tk_img = ImageTk.PhotoImage(img_PIL)
        cv2show.configure(image=tk_img)
        frame = cvtColor(np.array(img_PIL), COLOR_RGB2BGR)
        imshow('Output', frame)
        key = waitKey(detect_time)
        print(end_game)
        # if key == ord('q') or key == 27 or end_game or not start_flag:
        #     break

        # imgs = os.listdir('demo_video/')
        # for i in range(len(imgs)):
        #     frame = imread('demo_video/'+imgs[i])
        #     # frame = cv2.cvtColor(np.array(cap), cv2.COLOR_RGB2BGR)
        #     imshow('Output', frame)
        #     key = waitKey(detect_time)
        #     if key == ord('q') or key == 27 or end_game or not start_flag:
        #         break
    
    print('end run_model')

def start_game():
    global start_flag, start_time
    start_flag = True
    start_time = time.time()
    update_clock()

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
    print('close')
    print('endgame', end_game)
    time.sleep(1)
    app.destroy()

def update_clock():
    global start_time, start_flag

    delta = int(time.time() - start_time)
    current_time = str(timedelta(seconds=delta))
    
    if start_flag:
        my_time.config(text=current_time)
    else:
        my_time.config(text='0:00:00')

    app.after(1000, update_clock) 


t = threading.Thread(target = run_model)
t.start()

app = tk.Tk()

my_time = tk.Label(app, text = '0:00:00')
my_time.grid(column=0, row=0, ipadx=5, pady=5, sticky=tk.W+tk.N)

labelHeight = tk.Label(app, text = "Height Ratio")
labelHeight.grid(column=0, row=1, ipadx=5, pady=5, sticky=tk.W+tk.S)

start_button = tk.Button(app, text = 'Start', width=20, height=2, command=start_game)
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
cv2show = tk.Label(app, image=tmp)
cv2show.grid(column=0, row=4, pady=10, sticky=tk.W)

app.protocol("WM_DELETE_WINDOW", on_closing)

app.mainloop()

t.join()



# os.system("pause")

