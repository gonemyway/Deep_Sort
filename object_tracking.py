import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
from models.common import DetectMultiBackend, AutoShape

# Config value
video_path = "data.ext/highway.mp4"
conf_threshold = 0.5 # Tren 0.5 thi chap nhan ket qua predict cua model
tracking_class = 2 # So thu tu class can predict trong file classes.names (tracking tat ca thi de None)

# Khoi tao Deepsort
tracker = DeepSort(max_age=5)  # Sau 5 frame lien tiep ma khong phat hien vat the thi xoa vat the khoi bo nho

# Khoi tao yolov9
device = "cpu" # "cuda": GPU, "mps:0": macbook
model = DetectMultiBackend(weights="weights/yolov9-c-converted.pt", device=device, fuse=True)
model = AutoShape(model)

# Load classname tu file classes.names
with open("data.ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names), 3))
tracks = []

# Khoi tao VideoCapture de doc tu file video
cap = cv2.VideoCapture(video_path)

# Tien hanh doc tung frame tu video
while True:
    # Doc
    ret, frame = cap.read()
    if not ret:
        continue

    # Dua qua model de detect
    results = model(frame)
    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if tracking_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:
                continue

        detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])

    # Cap nhat, gan id bang DeepSort
    tracks = tracker.update_tracks(detect, frame=frame)

    # Ve len man hinh cac khung chu nhat kem id
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lay toa do, class_id de ve len hinh anh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hinh anh len man hinh
    cv2.imshow("OT", frame)

    # Bam Q thi thoat
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

