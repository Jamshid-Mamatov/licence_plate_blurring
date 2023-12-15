import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import torch

# from WPODNet_Pytorch.predict import *
from WPODNet_Pytorch.wpodnet.model import WPODNet
from WPODNet_Pytorch.wpodnet.backend import Predictor
from WPODNet_Pytorch.wpodnet.stream import ImageStreamer
# Load YOLOv8 model for car detection
model_car_detection = YOLO('yolov8s.pt')


# Load WPOD-NET model for license plate detection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = WPODNet()
model.to(device)

checkpoint = torch.load("wpodnet.pth")
model.load_state_dict(checkpoint)





def wpod_process(img):
    # Preprocess image for WPOD-NET
    # img = np.array(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (128, 64))
    # img = img / 255.0
    # img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=0)
    # img = torch.from_numpy(img).float()
    predictor = Predictor(model)

    prediction = predictor.predict(img, scaling_ratio=0.5)

    # print('Bounds:', prediction.bounds.tolist())
    # print('Confidence:', prediction.confidence)

    # print(prediction.bounds)
    # img.show()
    annotated = prediction.blur_polygon()
    # annotated.save(annotated_path)
    
    return annotated

cap = cv2.VideoCapture('20220906_122241_11.mp4')  # Replace 'your_input_video.mp4' with the actual video file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Detect cars using YOLOv8
    results = model_car_detection.predict(frame, classes=[2, 5, 7])

    for r in results:
        boxes = r.boxes.xyxy.numpy()
        for box in boxes:
            x, y, x2, y2 = map(int, box)
            
            # Crop the detected car
            car_roi = frame[y:y2, x:x2]

            # Detect license plates using WPOD-NET
            pillow_image = Image.fromarray(car_roi)

            lp_box = wpod_process(pillow_image)
            
            
            # Perform license plate detection using WPOD-NET
            # Note: You can directly use pillow_image here, no need to use ImageStreamer
            blurred = wpod_process(pillow_image)
            
            blurred.show()

            frame[y:y2, x:x2] = np.array(blurred)
           

        
            
    
    out.write(frame)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
