import cv2

# Configuration

with open('cocolabels.txt', 'r') as f:
    classNames = f.read().split('\n')

cgf_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights_file = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weights_file, cgf_file)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Handling image display and object detection


def display(keynum):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(255, 43, 244), thickness=3)
            cv2.putText(img, classNames[classId - 1], (box[0] + 10,
                                                       box[1] + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 43, 244), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(keynum)


choice = input('(img/webcam) ? : ')
choice.lower()

if choice == 'img':
    image = input('image file name: ')
    img = cv2.imread(f'{image}')
    display(0)
elif choice == 'webcam':
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        success, img = cap.read()
        display(1)
