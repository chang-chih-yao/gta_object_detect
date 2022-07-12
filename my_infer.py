from torch import hub
from cv2 import VideoCapture, rectangle, imshow, waitKey, imread, destroyAllWindows, putText, FONT_HERSHEY_SIMPLEX
import os
import time
import tkinter as tk
import threading
from PIL import ImageTk

detect_time = 400
start_flag = False
end_game = False

# model = hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
model = hub.load('chang-chih-yao/yolov5', 'custom', 'best.onnx', device='cpu')
model.conf = 0.3  # NMS confidence threshold
model.iou = 0.4  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 100  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference



def run_model():
    global start_flag, end_game

    while(True):

        if end_game:
            destroyAllWindows()
            break

        if not start_flag:
            destroyAllWindows()
            print('wait start')
            time.sleep(0.5)
            continue

        imgs = os.listdir('demo_video/')
        for i in range(len(imgs)):
            frame = imread('demo_video/'+imgs[i])
            # frame = cv2.cvtColor(np.array(cap), cv2.COLOR_RGB2BGR)
            start = time.time()
            results = model(frame, size=640)  # includes NMS
            bboxs = results.pandas().xyxy[0]
            
            if not bboxs.empty:
                # print(bboxs)
                bbox_E_area = 999999
                bbox_E = [0,0,0,0]
                for idx in range(len(bboxs.index)):
                    x1 = int(bboxs.iat[idx, 0])
                    y1 = int(bboxs.iat[idx, 1])
                    x2 = int(bboxs.iat[idx, 2])
                    y2 = int(bboxs.iat[idx, 3])
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
                
                if bbox_E_area != 999999:
                    rectangle(frame, (bbox_E[0], bbox_E[1]), (bbox_E[2], bbox_E[3]), (0, 0, 255), 2, 1)
            # print('===================================')
            fps = round(1/(time.time() - start), 1)
            # print(fps)
            putText(frame, str(fps), (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            imshow('Output', frame)
            key = waitKey(detect_time)
            if key == ord('q') or key == 27 or end_game or not start_flag:
                break
        
        


def start_game():
    global start_flag
    start_flag = True

def stop_game():
    global start_flag
    start_flag = False

def my_gui():
    global end_game

    app = tk.Tk()

    labelWidth = tk.Label(app, text = "Width Ratio")
    labelWidth.grid(column=0, row=0, ipadx=5, pady=5, sticky=tk.W+tk.N)

    labelHeight = tk.Label(app, text = "Height Ratio")
    labelHeight.grid(column=0, row=1, ipadx=5, pady=5, sticky=tk.W+tk.S)

    entryWidth = tk.Entry(app, width=20)
    entryHeight = tk.Entry(app, width=20)

    entryWidth.grid(column=1, row=0, padx=10, pady=5, sticky=tk.N)
    entryHeight.grid(column=1, row=1, padx=10, pady=5, sticky=tk.S)

    start_button = tk.Button(app, text = 'Start', command=start_game)
    start_button.grid(column=0, row=2, pady=10, sticky=tk.W)

    stop_button = tk.Button(app, text = 'Stop', command=stop_game)
    stop_button.grid(column=0, row=3, pady=10, sticky=tk.W)

    app.mainloop()

    end_game = True



if __name__ == '__main__':
    
    t = threading.Thread(target = my_gui)
    t.start()

    run_model()
    
    t.join()
    os.system("pause")

