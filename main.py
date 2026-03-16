import cv2
import json
import os
from detector import Detection
from analytics import Analytics
from ultralytics.utils.plotting import Annotator
from config.config import Input_imagePath, Output_imagePath

class Main:
    def __init__(self,):
        os.makedirs("./Output", exist_ok=True)
        self.detector = Detection()

    def ProcessImage(self,):
        detections = self.detector.detectObjects(Input_imagePath)
        analytics = Analytics(detections)

        img = cv2.imread(Input_imagePath)
        annotator = Annotator(img)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class"]
            if label in ["Person","vest","helmet"]:
                if label == "Person":
                    color = (255, 0, 0)    
                elif label == "helmet":
                    color = (52, 64, 235)      
                elif label == "vest":
                    color = (6,102,16)   
                annotator.box_label([x1,y1,x2,y2], label, color=color)
        result = annotator.result()
        cv2.imwrite(Output_imagePath, result)

        TotalWorkers_KPI = analytics.WorkersCount()
        WearingHelemt_or_not_KPI = list(analytics.Wearing_helmet_or_not())
        WearingVest_KPI = analytics.Wearning_Vest()
        Gender_KPI = list(analytics.GenderDetection(img))
        ImageQuality_KPI = analytics.ImageQuality(img)

        kpi_data = {
            "Total_Workers": TotalWorkers_KPI,
            "Wearing_Helmet": WearingHelemt_or_not_KPI[0],
            "Wearing_NoHelmet": WearingHelemt_or_not_KPI[1],
            "Helmet_Compliance_Rate(%)": WearingHelemt_or_not_KPI[2],
            "Wearing_Vest": WearingVest_KPI,
            "Male_Workers": Gender_KPI[0],
            "Female_Workers": Gender_KPI[1],
            "ImageQuality(%)": ImageQuality_KPI
        }

        with open("output/kpi_results.json", "w") as f:
            json.dump(kpi_data, f, indent=4)

if __name__ == "__main__":
    obj = Main()
    obj.ProcessImage()
