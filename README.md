# PPE KPI Detection

**Description:**  
Computer Vision project to detect construction site workers, helmets, and safety vests using YOLOv8. Generates key safety KPIs including total workers, helmet compliance, vest compliance, gender detection, and image quality. Annotated images and JSON results are automatically saved.

## Features
- Detects **Person, Helmet, and Vest** in images.
- Calculates KPIs: **Total Workers, Helmet & Vest Compliance, Gender Detection, Image Quality**.
- Annotates images with bounding boxes and labels.
- Saves **JSON files** with KPI results.
- Built with **YOLOv8, OpenCV, and DeepFace** for gender detection.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/Shivanchal-Asthana/PPE_KPI_Detection.git

**Create and activate virtual environment**
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

## Install dependencies:
pip install -r requirements.txt

## Usage
python main.py

NOTE:
Input image path is set in config/config.py
Output images and JSON KPI results are saved automatically

## Output
Annotated images with bounding boxes
JSON file containing KPIs:

{
  "TotalWorkers_KPI": 5,
  "WearingHelmet_or_not_KPI": [4, 1, 80.0],
  "WearingVest_KPI": 5,
  "Gender_KPI": [3, 2],
  "ImageQuality_KPI": 92.5
}
