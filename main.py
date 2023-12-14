import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
model_car_detection = YOLO('yolov8s.pt')

cap = cv2.VideoCapture('20220906_122241_11.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results=model_car_detection.predict(frame,classes=[2,5,7])
    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        # im.save('results.jpg')  # save image
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()