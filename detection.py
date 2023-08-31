from ultralytics import YOLO
import cv2
import numpy as np
from queue import Queue

def enqueue(que, data):
    if que.full():
        _ = que.get()
    que.put(data)


def xyxy_to_xyah(xyxy):
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    center_x = xyxy[0] + width/2
    center_y = xyxy[1] + height/2
    ret = np.asarray(xyxy).copy()
    ret[0] = center_x
    ret[1] = center_y
    ret[2] = width/height
    ret[3] = height

    return ret


def draw_line_trails(frame, bbox, que_trail_marks, color=(0, 0, 255)):
    annotated_image = frame.copy()
    x_center, y_center, *_ = xyxy_to_xyah(bbox).astype('int')
    for i in range(len(que_trail_marks.queue) - 1):
        cv2.line(annotated_image, que_trail_marks.queue[i][:2], que_trail_marks.queue[i+1][:2], color=(0, 0, 255), thickness=2)

    enqueue(que_trail_marks, (x_center, y_center))

    return annotated_image

def annotate(frame, bbox, color, method, thickness=2, conf=None, frame_num=0):
    annotated_image = frame.copy()

    cv2.putText(annotated_image, "Frame: " + str(frame_num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    
    if method == "":
        return annotated_image
    
    x1, y1, x2, y2 = bbox.astype('int')
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(annotated_image, conf, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    return annotated_image


if __name__ == "__main__":
    detector = YOLO("models/best_latest_441.pt")
    filename = "unseen-1.mp4"
    cap = cv2.VideoCapture(f"videos/{filename}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vidwriter = cv2.VideoWriter(f"kf-finetune-exp-results/Aug24/det-{filename}",
                                fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
                                fps=fps, 
                                frameSize=(int(width), int(height)), 
                                isColor=True
                            )
    que_trailmarks = Queue(maxsize=20)
   
    while True:
        ret, frame = cap.read()
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print("Frame: ", frame_num)
        if not ret:
            break

        detections = detector.predict(frame, conf=0.4, max_det=3, device=0, verbose=False)
        measurement = detections[0].boxes.xyxy.cpu().numpy()
        conf = detections[0].boxes.conf.cpu().numpy()

        isObjectDetected = True if len(measurement) > 0 else False

        if isObjectDetected:
            annotated_frame = annotate(frame, measurement[0], color=(0, 0, 255), method="detect", thickness=2, frame_num=frame_num)
            annotated_frame = draw_line_trails(annotated_frame, measurement[0], que_trailmarks, color=(0, 0, 255))
            vidwriter.write(annotated_frame)
        else:
            annotated_frame = annotate(frame, measurement, color=(0, 0, 255), method="", thickness=2, frame_num=frame_num)
            vidwriter.write(annotated_frame)

