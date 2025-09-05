import cv2
import winsound  # for sound alert on Windows
from ultralytics import YOLO

#loading trained  to modelobject
model = YOLO("runs/detect/train/weights/best.pt")

#jo camera select krna hai
cam_index = 0
cap = cv2.VideoCapture(cam_index)

# Make window resizable
cv2.namedWindow("Fire Detection by Ankit", cv2.WINDOW_NORMAL)

# Recording setup
record = False
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, conf=0.5, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Check for fire/smoke detection
    if len(results[0].boxes) > 0:  # if objects detected
        for box in results[0].boxes:
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])  # confidence score
            if conf > 0.5:  
                # Play sound when fire/smoke detected
                winsound.Beep(1000, 200)  # frequency=1000Hz, duration=200ms
                break  # play once per frame

    # Show live annotated feed
    cv2.imshow("Fire Detection by Ankit, shivam , nitesh", annotated_frame)

    # If recording enabled, save frame
    if record and out is not None:
        out.write(annotated_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # quit
        break
    elif key == ord("r"):  # toggle recording
        if not record:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("fire_webcam_output.mp4", fourcc, 20.0,
                                  (annotated_frame.shape[1], annotated_frame.shape[0]))
            record = True
            print("Recording started...")
        else:
            record = False
            out.release()
            out = None
            print("Recording stopped.")
    elif key == ord("c"):  # switch camera
        cam_index = (cam_index + 1) % 3  # cycle between camera 0,1,2
        cap.release()
        cap = cv2.VideoCapture(cam_index)
        print(f"Switched to camera {cam_index}")

# Cleanup
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
