
from ultralytics import YOLO

# Load trained model
model = YOLO("../models/best.pt")

def predict(image_path):
    results = model.predict(image_path, save=True, conf=0.25)

    for r in results:
        print(r.boxes)

# Example test
predict("test.jpg")
