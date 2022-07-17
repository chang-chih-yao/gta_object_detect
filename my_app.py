import os
import time
import tkinter as tk
from threading import Thread
from cv2 import VideoCapture, rectangle, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_RGB2BGR, COLOR_BGR2RGB, TrackerCSRT_create, resize, INTER_AREA
from PIL import ImageTk, Image
import pydirectinput as pd
from pyautogui import screenshot, moveRel
import pygetwindow as gw
import numpy as np
from datetime import timedelta
from math import sqrt


from torch import hub
# model = hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
model = hub.load('chang-chih-yao/yolov5', 'custom', 'best.onnx', device='cpu')
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.4  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 100  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference

tracker = TrackerCSRT_create()

detect_time = 82
start_time = 0   # for GUI timer
start_flag = False
end_game = False
cv_enable = True
e_trigger = False


print(gw.getAllTitles())
# ttt = gw.getWindowsWithTitle('Visual')[0]
# while(True):
#     time.sleep(1)
#     print(ttt.isActive)
#     break

gta_ = gw.getWindowsWithTitle('FiveM')[0]

offset = 10
left = gta_.left + offset
top = gta_.top + offset+20
w = 1280
h = 720
print(left, top, w, h)

human_x1 = 563
human_x2 = 617
human_y1 = 325
human_y2 = 530

target_x1 = 0
target_x2 = 0
target_y1 = 0
target_y2 = 0

img_PIL = screenshot(region=(left, top, w, h))
tk_img = ''


# sight_move_range = 300
# sight_move_speed = 0.5
# def sight_l():
#     moveRel(-sight_move_range, 0, duration = sight_move_speed)

class my_keyboard:
    def __init__(self):
        self.w_key = False
        self.a_key = False
        self.s_key = False
        self.d_key = False

    # def press_e(self):
    #     self.release_all()
    #     print('press e')
    #     pd.press('e')

    def l(self):
        global a_label
        if not self.a_key:
            self.release_left_right()
            self.a_key = True
            print('go left')
            pd.keyDown('a')
            a_label.configure(bg='#FF0')

    def l_(self):
        global a_label
        self.release_left_right()
        print('go left_')
        pd.keyDown('a')
        a_label.configure(bg='#FF0')
        time.sleep(0.2)
        pd.keyUp('a')
        a_label.configure(bg='#FFF')
        time.sleep(0.2)

    def r(self):
        global d_label
        if not self.d_key:
            self.release_left_right()
            self.d_key = True
            print('go right')
            pd.keyDown('d')
            d_label.configure(bg='#FF0')

    def r_(self):
        global d_label
        self.release_left_right()
        print('go right_')
        pd.keyDown('d')
        d_label.configure(bg='#FF0')
        time.sleep(0.2)
        pd.keyUp('d')
        d_label.configure(bg='#FFF')
        time.sleep(0.2)

    def up(self):
        global w_label
        if not self.w_key:
            # self.release_up()
            self.w_key = True
            print('go up')
            pd.keyDown('w')
            w_label.configure(bg='#FF0')

    def release_up(self):
        global w_label
        if self.w_key:
            self.w_key = False
            pd.keyUp('w')
            w_label.configure(bg='#FFF')

    def release_left_right(self):
        global a_label, d_label
        if self.a_key:
            self.a_key = False
            pd.keyUp('a')
            a_label.configure(bg='#FFF')
        if self.d_key:
            self.d_key = False
            pd.keyUp('d')
            d_label.configure(bg='#FFF')

    def release_all(self):
        global w_label, a_label, d_label
        if self.w_key:
            self.w_key = False
            pd.keyUp('w')
            w_label.configure(bg='#FFF')
        if self.a_key:
            self.a_key = False
            pd.keyUp('a')
            a_label.configure(bg='#FFF')
        if self.s_key:
            self.s_key = False
            pd.keyUp('s')
        if self.d_key:
            self.d_key = False
            pd.keyUp('d')
            d_label.configure(bg='#FFF')

ky = my_keyboard()

def keyboard_ctrl():
    global target_x1, target_x2, target_y1, target_y2, e_trigger
    while(True):
        if end_game:
            print('ready to end game keyboard')
            break
        
        if not start_flag:
            time.sleep(0.5)
            continue

        wait_time = int(detect_time/2)
        time.sleep(wait_time/1000.0)

        # if e_trigger:                               # 偵測到採集時
        #     ky.press_e()
        #     time.sleep(0.5)
        if target_x1 == -1 and target_y1 == -1 and target_x2 == -1 and target_y2 == -1:   # 沒偵測到任何東西
            # sight_l()
            ky.l()
            ky.release_up()
        else:
            ky.up()
            human_x = (human_x1 + human_x2)/2
            human_y = (human_y1 + human_y2)/2
            target_x = (target_x1 + target_x2)/2
            target_y = (target_y1 + target_y2)/2

            if human_x >= target_x:         # target在人物左邊
                if target_x2 <= human_x1:   # 可往左邊走
                    if (human_x1 - target_x2) > 35:
                        ky.l()
                    elif (human_x1 - target_x2) > 10:
                        ky.l_()
                else:                       # 重疊了
                    if (target_x2 - human_x1) > 20:
                        ky.r_()
            else:                           # target在人物右邊
                if human_x2 <= target_x1:   # 可往右邊走
                    if (target_x1 - human_x2) > 35:
                        ky.r()
                    elif (target_x1 - human_x2) > 10:
                        ky.r_()
                else:                       # 重疊了
                    if (human_x2 - target_x1) > 20:
                        ky.l_()

def run_model():
    print('in run_model')
    global img_PIL, tk_img, imgshow, detect_time, cv_enable, target_x1, target_x2, target_y1, target_y2, e_trigger

    cou = 0
    tracking_bbox = (10, 10, 100, 100)    # (x1, y1, w, h)
    tracker = TrackerCSRT_create()
    tracker.init(cvtColor(np.array(img_PIL), COLOR_RGB2BGR), tracking_bbox)

    while(True):
        if end_game:
            print('ready to end game')
            break
        
        if not start_flag:
            print('wait start')
            time.sleep(0.5)
            continue

        start = time.time()

        

        img_PIL = screenshot(region=(left, top, w, h))
        frame = cvtColor(np.array(img_PIL), COLOR_RGB2BGR)
        track_img = frame.copy()

        
        
        rectangle(frame, (human_x1, human_y1), (human_x2, human_y2), (0, 255, 0), 2, 1)

        if cou%6 == 0:
            results = model(img_PIL, size=640)  # includes NMS
            bboxs = results.pandas().xyxy[0]

            human_x = (human_x1 + human_x2)/2
            human_y = (human_y1 + human_y2)/2

            tracking_bbox_size = (tracking_bbox[2] + tracking_bbox[3])/2
            tracking_bbox_center_x = tracking_bbox[0] + tracking_bbox[2]/2
            tracking_bbox_center_y = tracking_bbox[1] + tracking_bbox[3]/2
            
            min_dist = 9999999
            min_dist_tracking = 9999999
            target_x1_tmp = 0
            target_x2_tmp = 0
            target_y1_tmp = 0
            target_y2_tmp = 0
            # e_trigger = False
            
            
            if not bboxs.empty:
                # print(bboxs)
                bbox_E_area = 999999
                bbox_E = [0,0,0,0]
                for idx in range(len(bboxs.index)):
                    x1 = int(bboxs.iat[idx, 0])
                    y1 = int(bboxs.iat[idx, 1])
                    x2 = int(bboxs.iat[idx, 2])
                    y2 = int(bboxs.iat[idx, 3])
                    yolo_bbox_center_x = (x1+x2)/2
                    yolo_bbox_center_y = (y1+y2)/2
                    conf = float(bboxs.iat[idx, 4])
                    obj_name = bboxs.iat[idx, 6]
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    area = bbox_w*bbox_h
                    # print(area)
                    # convertedImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if obj_name == 'E':
                        if area < bbox_E_area:
                            bbox_E_area = area
                            bbox_E = [x1, y1, x2, y2]
                    elif area > 1000:
                        rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, 1)
                        distence = sqrt((yolo_bbox_center_x - human_x)**2 + (yolo_bbox_center_y - human_y)**2)
                        if distence < min_dist:   # 選出距離 human 最近的 yolo bbox
                            min_dist = distence
                            target_x1_tmp = x1
                            target_x2_tmp = x2
                            target_y1_tmp = y1
                            target_y2_tmp = y2

                        distence_tracking = sqrt((yolo_bbox_center_x - tracking_bbox_center_x)**2 + (yolo_bbox_center_y - tracking_bbox_center_y)**2)
                        if distence_tracking < min_dist_tracking:   # 選出距離 tracking bbox 最近的 yolo bbox
                            min_dist_tracking = distence_tracking
                            tracking_bbox = (x1, y1, x2-x1, y2-y1)
                
                if bbox_E_area != 999999:
                    # e_trigger = True
                    rectangle(frame, (bbox_E[0], bbox_E[1]), (bbox_E[2], bbox_E[3]), (0, 0, 255), 2, 1)

                if target_x1_tmp == 0 and target_y1_tmp == 0 and target_x2_tmp == 0 and target_y2_tmp == 0:  # 若沒偵測到東西
                    target_x1 = -1
                    target_x2 = -1
                    target_y1 = -1
                    target_y2 = -1
                else:
                    if min_dist_tracking < tracking_bbox_size:     # 若 tracking bbox 跟最近的 yolo bbox 的距離小於 tracking bbox 的大小，就可以用 yolo bbox 更新 tracking bbox 的位置
                        tracker = TrackerCSRT_create()
                        tracker.init(track_img, tracking_bbox)
                        # print('tracking box update by yolo')

                        target_x1 = tracking_bbox[0]
                        target_x2 = tracking_bbox[0] + tracking_bbox[2]
                        target_y1 = tracking_bbox[1]
                        target_y2 = tracking_bbox[1] + tracking_bbox[3]
                        rectangle(frame, (target_x1, target_y1), (target_x2, target_y2), (255, 255, 0), 2, 1)
                    else:                                          # tracking 更新到全新的位置
                        tracker = TrackerCSRT_create()
                        tracker.init(track_img, (target_x1_tmp, target_y1_tmp, target_x2_tmp-target_x1_tmp, target_y2_tmp-target_y1_tmp))
                        # print(track_img.size, target_x1_tmp, target_y1_tmp, target_x2_tmp, target_y2_tmp)
                        target_x1 = target_x1_tmp
                        target_x2 = target_x2_tmp
                        target_y1 = target_y1_tmp
                        target_y2 = target_y2_tmp
                        rectangle(frame, (target_x1, target_y1), (target_x2, target_y2), (255, 255, 0), 2, 1)

                    # if cou == 0:    # begining
                    #     tracking_bbox = (target_x1, target_y1, target_x2-target_x1, target_y2-target_y1)
                    #     tracker.init(track_img, tracking_bbox)
            else:
                target_x1 = -1
                target_x2 = -1
                target_y1 = -1
                target_y2 = -1
        
        else:
            if target_x1 == -1 and target_x2 == -1 and target_y1 == -1 and target_y2 == -1:   # 沒偵測到任何東西
                pass
            else:
                ok, tracking_bbox = tracker.update(track_img)
                if ok:
                    p1 = (int(tracking_bbox[0]), int(tracking_bbox[1]))
                    p2 = (int(tracking_bbox[0] + tracking_bbox[2]), int(tracking_bbox[1] + tracking_bbox[3]))
                    target_x1 = int(tracking_bbox[0])
                    target_y1 = int(tracking_bbox[1])
                    target_x2 = int(tracking_bbox[0] + tracking_bbox[2])
                    target_y2 = int(tracking_bbox[1] + tracking_bbox[3])

                    rectangle(frame, p1, p2, (255,255,0), 2, 1)
        
        # print('===================================')
        fps = round(1/(time.time() - start), 1)
        # print(fps)
        putText(frame, str(fps), (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



        

        if cv_enable:
            putText(frame, str(fps), (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            frame_resize = resize(frame, (640, 360), interpolation=INTER_AREA)
            img_PIL = Image.fromarray(cvtColor(frame_resize, COLOR_BGR2RGB))
            tk_img = ImageTk.PhotoImage(img_PIL)
            imgshow.configure(image=tk_img)

        cou += 1
        time.sleep(detect_time/1000.0)

    




def speed_low():
    global detect_time, speed_lab
    detect_time += 20
    speed_lab.configure(text='infer speed : ' + str(detect_time))

def speed_fast():
    global detect_time, speed_lab
    if detect_time >= 42:
        detect_time -= 20
    speed_lab.configure(text='infer speed : ' + str(detect_time))

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
        my_time.config(text='已耗時: '+current_time)
    else:
        my_time.config(text='已耗時: 0:00:00')

    app.after(1000, update_clock) 


class task:
    def __init__(self):
        self.lock = False

    def newThread(self, my_str):
        if self.lock == True:
            print('some thread running...')
            while(self.lock):
                time.sleep(0.2)
            print('OK!')

        if self.lock == False:
            if my_str == 'img_on_off':
                self.lock = True
                Thread(target=self.img_on_off).start()
            elif my_str == 'start_game':
                self.lock = True
                Thread(target=self.start_game).start()
            elif my_str == 'stop_game':
                self.lock = True
                Thread(target=self.stop_game).start()
            else:
                print('wrong cmd!')
                exit()
            
    
    def img_on_off(self):
        global cv_enable, img_on_off_label
        print(cv_enable)
        if cv_enable:
            cv_enable = False
            img_on_off_label.configure(text='顯示畫面 : OFF')
        else:
            cv_enable = True
            img_on_off_label.configure(text='顯示畫面 : ON')
        self.lock = False

    def start_game(self):
        global start_flag, start_time, gta_, left, top, w, h, labelHeight
        if not start_flag:
            gta_ = gw.getWindowsWithTitle('FiveM')[0]

            left = gta_.left + offset
            top = gta_.top + offset+20
            w = 1280
            h = 720
            print(left, top, w, h)

            gta_.activate()
            time.sleep(0.5)
            start_flag = True
            labelHeight.configure(text='狀態 : 採礦中')
            start_time = time.time()
            update_clock()
        self.lock = False

    def stop_game(self):
        global start_flag, labelHeight
        if start_flag:
            start_flag = False
            time.sleep(0.5)
            ky.release_all()
            labelHeight.configure(text='狀態 : 停止')
        self.lock = False


op = task()

def thread_func(my_str):
    op.newThread(my_str)


def detect_focus():
    time.sleep(2)
    while(True):
        time.sleep(0.5)
        if end_game:
            print('ready to end game')
            break
        
        if gta_.isActive == False and start_flag:
            op.newThread('stop_game')

def eee():
    while(True):
        if end_game:
            print('ready to end game eee')
            break
        
        if not start_flag:
            time.sleep(0.5)
            continue

        pd.press('e')
        time.sleep(0.15)

########################### main loop ###########################

t = Thread(target = run_model)
t2 = Thread(target=keyboard_ctrl)
t3 = Thread(target=detect_focus)
t4 = Thread(target=eee)


app = tk.Tk()
app.title('MY_GTA')
app.geometry('750x650+5+5')

my_time = tk.Label(app, text = '已耗時 : 0:00:00', width=15, height=2)
my_time.place(x=10, y=10)

w_label = tk.Label(app, text = 'W', width=5, height=2, bg='#FFF')
w_label.place(x=210, y=10)

labelHeight = tk.Label(app, text = '狀態 : 停止', width=15, height=2)
labelHeight.place(x=10, y=60)

a_label = tk.Label(app, text = 'A', width=5, height=2, bg='#FFF')
a_label.place(x=160, y=60)

d_label = tk.Label(app, text = 'D', width=5, height=2, bg='#FFF')
d_label.place(x=260, y=60)

start_button = tk.Button(app, text = 'Start', width=15, height=2, command=lambda: thread_func('start_game'))
start_button.place(x=10, y=110)

# stop_button = tk.Button(app, text = 'Stop', width=15, height=2, command=lambda: thread_func('stop_game'))
# stop_button.grid(column=1, row=2, pady=10, sticky=tk.W)

speed_slow_btn = tk.Button(app, text = 'slow', width=15, height=2, command=speed_low)
speed_slow_btn.place(x=10, y=160)

speed_lab = tk.Label(app, text = 'infer speed : ' + str(detect_time))
speed_lab.place(x=160, y=170)

speed_fast_btn = tk.Button(app, text = 'fast', width=15, height=2, command=speed_fast)
speed_fast_btn.place(x=310, y=160)

img_on_off_btn = tk.Button(app, text = 'ON/OFF', width=15, height=2, command=lambda: thread_func('img_on_off'))
img_on_off_btn.place(x=10, y=210)

img_on_off_label = tk.Label(app, text = '顯示畫面 : ON', width=15, height=2)
img_on_off_label.place(x=160, y=220)

new_img = img_PIL.resize((640, 360))
tmp = ImageTk.PhotoImage(new_img)
imgshow = tk.Label(app, image=tmp)
imgshow.place(x=10, y=260)



app.protocol("WM_DELETE_WINDOW", Thread(target=on_closing).start)



t.start()
t2.start()
t3.start()
t4.start()
app.mainloop()
print('app end')

t.join()
print('while end')
t2.join()
print('keyboard end')
t3.join()

# sys.exit()



# os.system("pause")

