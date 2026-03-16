from deepface import DeepFace
from ultralytics.utils.plotting import Annotator
import cv2
from config.config import Output_imagePath1

class Analytics:
    def __init__(self, detections):
        self.detections = detections

    def ImageQuality(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        max_score = 1000  
        quality_percent = min((total_score / max_score) * 100, 100)
        return round(quality_percent,2)

    def WorkersCount(self,):
        TotalWorker = [i for i in self.detections if i['class'] == "Person"]
        return len(TotalWorker)
    
    def Wearing_helmet_or_not(self,):
        HelemtCount = [i for i in self.detections if i['class'] == "helmet"]
        NoHelemtCount = self.WorkersCount() - len(HelemtCount)
        HelmetComplainceRate = round((len(HelemtCount)/self.WorkersCount())*100,2)
        return len(HelemtCount), NoHelemtCount, HelmetComplainceRate
    
    def Wearning_Vest(self,):
        vestCount = [i for i in self.detections if i['class'] == "vest"]
        return len(vestCount)
    
    def GenderDetection(self, frame):
        male = 0
        female = 0

        self.annotator = Annotator(frame, line_width=4)
        persons = [i for i in self.detections if i['class'] == "Person"]

        for idx, p in enumerate(persons):
            x1, y1, x2, y2 = p['bbox']

            width = x2 - x1
            height = y2 - y1

            cx = x1 + width // 2
            new_w = int(width * 0.6)

            new_x1 = int(cx - new_w / 2)
            new_x2 = int(cx + new_w / 2)
            head_crop = frame[y1:y1 + int(height * 0.5), new_x1:new_x2]

            if head_crop.size == 0:
                continue

            try:
                result = DeepFace.analyze(head_crop,actions=['gender'],enforce_detection=False)
                gender = result[0]['dominant_gender']

                if gender == "Man":
                    male += 1
                    label = "Male"
                    color = (0,0,0)
                else:
                    female += 1
                    label = "Female"
                    color = (255,0,255)

                self.annotator.box_label((x1, y1, x2, y2),label,color=color)
            except:
                pass

        frame = self.annotator.result()
        cv2.imwrite(Output_imagePath1, frame)

        return male, female
    
    
    
    


               

