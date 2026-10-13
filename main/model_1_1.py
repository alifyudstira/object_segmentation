import cv2
import onnxruntime as ort
from ultralytics import YOLO

model=YOLO("onnx\segmentation_model_1_1.onnx")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam gagal dibuka...")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8-seg", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()