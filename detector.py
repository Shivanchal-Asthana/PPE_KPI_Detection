from ultralytics import YOLO
import cv2
from config.config import modelPath

class Detection:
    def __init__(self,):
        self.model = YOLO(modelPath, verbose=False)
        self.detections = []

    def detectObjects(self,imagePath):
        img = cv2.imread(imagePath)

        result = self.model(img, verbose=False)[0]

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.detections.append({"class": label,
                                    "bbox": [x1, y1, x2, y2],
                                    "conf": conf})
        return self.detections






    





