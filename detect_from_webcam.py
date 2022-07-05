import cv2
from imageai.Detection import ObjectDetection
import numpy as np
import requests as req
import os as os
import random


# url = 'https://p7.hiclipart.com/preview/124/937/193/architectural-engineering-engineer.jpg'
# r = req.get(url)
# with open('testimage.jpg', 'wb') as outfile:
#     outfile.write(r.content)


modelRetinaNet = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5'
modelYOLOv3 = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5'
modelTinyYOLOv3 = 'https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5'

if not os.path.exists('yolo.h5'):
    r = req.get(modelYOLOv3, timeout=2)
    with open('yolo.h5', 'wb') as outfile:
        outfile.write(r.content)


detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('yolo.h5')
detector.loadModel()

# peopleImages = os.listdir("people")
# randomFile = peopleImages[random.randint(0, len(peopleImages) - 1)]

peopleOnly = detector.CustomObjects(person=True)


cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    cv2.imwrite('output.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # detectedImage, detections = detector.detectCustomObjectsFromImage(custom_objects=peopleOnly, output_type="array", input_image='output.jpg', minimum_percentage_probability=30)
    detectedImage, detections = detector.detectObjectsFromImage(output_type="array", input_image='output.jpg', minimum_percentage_probability=30)
    convertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)

    # for eachObject in detections:
    #     print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )

    cv2.imshow('Output', convertedImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

'''
detectedImage, detections = detector.detectCustomObjectsFromImage(custom_objects=peopleOnly, output_type="array", input_image="people/exwait.jpg", minimum_percentage_probability=30)
convertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)

# detectedImage, detections = detector.detectObjectsFromImage(output_type="array", input_image="people/exwait.jpg", minimum_percentage_probability=30)
# convertedImage = cv2.cvtColor(detectedImage, cv2.COLOR_RGB2BGR)


for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")

cv2.imshow('Output', convertedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








'''
from imageai.Detection.Custom import DetectionModelTrainer


trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="hardhat")

trainer.setTrainConfig(object_names_array=["person hardhat"], batch_size=4, num_experiments=20, 
                       train_from_pretrained_model="yolo.h5")

# trainer.trainModel()


model05 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-005--loss-0013.983.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model10 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-010--loss-0010.968.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model15 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-015--loss-0009.434.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)
model18 = trainer.evaluateModel(model_path="hardhat\models\detection_model-ex-018--loss-0008.717.h5", 
                      json_path="hardhat\json\detection_config.json", iou_threshold=0.5, 
                      object_threshold=0.3, nms_threshold=0.5)

print('---------------------------------------------------------')
print('Iteration 05:', model05[0]['average_precision']['person hardhat'])
print('Iteration 10:', model10[0]['average_precision']['person hardhat'])
print('Iteration 15:', model15[0]['average_precision']['person hardhat'])
print('Iteration 18:', model18[0]['average_precision']['person hardhat'])
print('---------------------------------------------------------')

'''