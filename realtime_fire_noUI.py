from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run real-time detection from webcam
model.predict(
    source=0,        # 0 = default webcam
    show=True,       # show output window
    conf=0.5,        # confidence threshold
    project="results",
    name="fire_webcam"
)
