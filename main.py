import cv2

# Configuration

with open('cocolabels.txt', 'r') as f:
    classNames = f.read().split('\n')

cgf_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_file = 'frozen_inference_graph.pb'


# Handling image display and object detection

image = input('image file name: ')
img = cv2.imread(f'{image}')

net = cv2.dnn_DetectionModel(weights_file, cgf_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)

print(classIds, bbox)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(255, 43, 244), thickness=3)
    cv2.putText(img, classNames[classId - 1], (box[0] + 10,
                                               box[1] + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 43, 244), 2)


cv2.imshow("Output", img)
cv2.waitKey(0)
