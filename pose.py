from ultralytics import YOLO
import os

# Load YOLO pose model
model = YOLO("yolo11l-pose.pt")

# Directory of your training images
directory_path = "./images"

# Loop through images
for entry in os.listdir(directory_path):
    img_path = os.path.join(directory_path, entry)
    print(img_path)
    # Run pose prediction
    results = model.predict(source=img_path, save=True, show=False)
    print(f"{img_path} is annotated")
