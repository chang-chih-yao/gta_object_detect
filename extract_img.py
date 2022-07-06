import cv2
import os as os
import shutil

video_name = 'rock_1.mkv'
extract_dir = 'extract/'
dataset_dir = 'dataset/'
extract_img_dir = extract_dir + video_name.split('.')[0] + '/'
train_dir = dataset_dir + video_name.split('.')[0] + '/train/images/'
val_dir = dataset_dir + video_name.split('.')[0] + '/validation/images/'
val_num = 8   # 1/8 data for validation

# if(os.path.isdir(extract_img_dir)):
#     print('extract_dir already exists!!')
#     c = input('continue? (y/n) ')
#     if c.lower() == 'n':
#         exit()
#     else:
#         shutil.rmtree(extract_img_dir)

# os.mkdir(extract_img_dir)

# vidcap = cv2.VideoCapture(video_name)
# success, image = vidcap.read()
# count = 0
# img_cnt = 0
# while success:
#     if count%5 == 0:
#         cv2.imwrite(extract_img_dir + '{:0>5d}.jpg'.format(img_cnt), image)     # save frame as JPEG file      
#         print('save img:', img_cnt)
#         img_cnt += 1
    
#     success, image = vidcap.read()
#     count += 1


if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(dataset_dir + video_name.split('.')[0] + '/train/annotations/')
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    os.makedirs(dataset_dir + video_name.split('.')[0] + '/validation/annotations/')

extract_img = os.listdir(extract_img_dir)

for i in range(0, len(extract_img)):
    file = extract_img_dir + extract_img[i]
    if os.path.isfile(file):
        if i%val_num != 0:
            os.rename(file, train_dir + extract_img[i])
        else:
            os.rename(file, val_dir + extract_img[i])
    

