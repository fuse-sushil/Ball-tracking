from kalman_filter import KalmanFilter
from ultralytics import YOLO
import cv2
import numpy as np
from queue import Queue
from collections import defaultdict
import json

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

def xyah_to_xyxy(xyah): 
    width = xyah[2] * xyah[3]
    x_min = xyah[0] - width/2
    y_min = xyah[1] - xyah[3]/2
    ret = np.asarray(xyah).copy()
    ret[0] = x_min
    ret[1] = y_min
    ret[2] = x_min + width
    ret[3] = y_min + xyah[3]

    return ret

def annotate(frame, bbox, color, method, thickness=2, conf=None, frame_num=0):
    annotated_image = frame.copy()
    x1, y1, x2, y2 = bbox.astype('int')
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness=2)
    if method == "detect":
        cv2.putText(annotated_image, "Detect", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    elif method == "kalman":
        cv2.putText(annotated_image, "Estimate", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    elif method == "interpolate":
        cv2.putText(annotated_image, "Interpolate", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    
    cv2.putText(annotated_image, "Frame: " + str(frame_num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=thickness)
    return annotated_image

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_ball_in_play(measurements, ball_in_play):
    x_center_ball, y_center_ball, *_ = xyxy_to_xyah(ball_in_play)
    detections = list(map(lambda det: xyxy_to_xyah(det), measurements))
    distances = [euclidean_distance((det[0], det[1]), (x_center_ball, y_center_ball)) for det in detections]
    idx = np.argmin(distances)

    return measurements[idx]

def enqueue(que, data):
    if que.full():
        _ = que.get()
    que.put(data)

def draw_trails(frame, bbox, que_trail_marks, color=(0, 0, 255)):
    annotated_image = frame.copy()
    x_center, y_center, *_ = xyxy_to_xyah(bbox).astype('int')
    for (x_cen, y_cen, col) in list(que_trail_marks.queue):
        cv2.circle(annotated_image, (x_cen, y_cen), 3, col, -1)
    enqueue(que_trail_marks, (x_center, y_center, color))

    return annotated_image

def draw_line_trails(frame, bbox, que_trail_marks, color=(0, 0, 255)):
    annotated_image = frame.copy()
    x_center, y_center, *_ = xyxy_to_xyah(bbox).astype('int')
    for i in range(len(que_trail_marks.queue) - 1):
        cv2.line(annotated_image, que_trail_marks.queue[i][:2], que_trail_marks.queue[i+1][:2], color=(0, 0, 255), thickness=2)
    enqueue(que_trail_marks, (x_center, y_center, color))

    return annotated_image

def get_xyah_interpolation_model(que_past, que_future):
    for measure, frame_n in que_past.queue:
        t.append(frame_n)
        x.append(measure[0])
        y.append(measure[1])
        a.append(measure[2])
        h.append(measure[3])
    
    for measure, frame_n, _, _ in que_future.queue:
        t.append(frame_n)
        x.append(measure[0])
        y.append(measure[1])
        a.append(measure[2])
        h.append(measure[3])
    
    x_model = np.poly1d(np.polyfit(t, x, 2))
    y_model = np.poly1d(np.polyfit(t, y, 2))
    a_model = np.poly1d(np.polyfit(t, a, 2))
    h_model = np.poly1d(np.polyfit(t, h, 2))

    return x_model, y_model, a_model, h_model

def interpolate_missed_frames(x_model, y_model, a_model, h_model, missed_frames):
    interpolated_missed_frames = []
    for frame_missed, frame_n in missed_frames:
        x_interpolate = x_model(frame_n)
        y_interpolate = y_model(frame_n)
        a_interpolate = a_model(frame_n)
        h_interpolate = h_model(frame_n)
        interpolated_missed_frames.append(([x_interpolate, y_interpolate, a_interpolate, h_interpolate],
                                        frame_n,
                                        frame_missed,
                                        "missed"))
    return interpolated_missed_frames

def process_buffered_frames(mean,variance, held_frames, que_future, que_trailmarks, vidwriter):
    for i, (measure, frame_n, frame_held, tag) in enumerate(held_frames):
        mean, variance = kf.predict(mean, variance) 
        mean , variance = kf.update(mean, variance, measure)
        
        enqueue(que=que_past, data=(measure, frame_n))
        method = "detect" if tag == "future" else "kalman"
        
        if frame_n > que_future.queue[-1][1]:
            method = "supress_extrapolate"

        color = (0, 0, 255) if method == "detect" else (0, 0, 0)
        
        if method == "detect":
            annotated_frame = annotate(frame_held, xyah_to_xyxy(measure), color=color, method=method, frame_num=frame_n)
            annotated_frame = draw_trails(annotated_frame, xyah_to_xyxy(measure), que_trailmarks, color=color)
            print("Frame: ", frame_n, "=> rendered")
        elif method == "kalman":
            annotated_frame = annotate(frame_held, xyah_to_xyxy(mean[0:4]), color=color, method=method, frame_num=frame_n)
            annotated_frame = draw_trails(annotated_frame, xyah_to_xyxy(mean[0:4]), que_trailmarks, color=color)
            print("Frame: ", frame_n, "=> rendered")
        elif method == "supress_extrapolate":
            annotated_frame = frame_held.copy()
            # cv2.imwrite(f"missed_frames/{filename}_{frame_n}.jpg", annotated_frame)  
            cv2.putText(annotated_frame, "Frame: " + str(frame_n), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)
            
        vidwriter.write(annotated_frame)

    return mean, variance

if __name__ == '__main__':
    detector = YOLO("../models/best_latest_557.pt")
    filename = "unseen-5.mp4"

    cap = cv2.VideoCapture(f"../videos/{filename}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)   )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS: ",fps, "Width: ", width, "Height: ", height)
    print("Total number of frames: ", total_frames)
    vidwriter = cv2.VideoWriter(f"../kf-finetune-exp-results/Aug30/{filename}",
                                fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
                                fps=fps, 
                                frameSize=(int(width), int(height)), 
                                isColor=True
                            )

    que_past = Queue(maxsize=4)
    que_future = Queue(maxsize=3)
    que_trailmarks = Queue(maxsize=20)
    kf = KalmanFilter()
    mean = np.zeros((1,12))
    variance = np.eye(12)

    isTrackInitialised = False
    isObjectedDetected = False
    isBufferOn = False
    isMultipleObjectsDetected = False
    isAfterReset = False

    missed_frames = []
    interpolated_missed_frames = []
    x, y, a, h, t = [], [], [], [], []

    # missed_detection_logs = defaultdict(list)
    MAX_DISTANCE_PER_FRAME = 100  # 100
    MAX_NUM_CONSECUTIVE_MISS = int(fps)  # 1 sec

    while True:
        ret, frame = cap.read()
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if not ret:
            break
        
        detections = detector.predict(frame, conf=0.5, max_det=3, device=0, verbose=False)  
        measurement = detections[0].boxes.xyxy.cpu().numpy()
        # conf = detections[0].boxes.conf.cpu().numpy()
        isObjectDetected = True if len(measurement) > 0 else False
        isMultipleObjectsDetected = True if len(measurement) > 1 else False
        
        # Debugging block
        # isAfterReset = False    
        print("="*50)
        print("Current Frame: ", frame_num)
        print("Ball detected: ", isObjectDetected)
        if missed_frames != []:
            print("missed frame", np.array(missed_frames, dtype=object)[:, 1])
        if list(que_future.queue) != []:
            print("future frames: ", np.array(list(que_future.queue), dtype=object)[:, 1])
        if list(que_past.queue) != []:
            print("past frames: ", np.array(list(que_past.queue), dtype=object)[:, 1])


        if isTrackInitialised and isObjectDetected:
            # For self adjusting distance threshold to account for ball acceleration and make it trajectory independent
            temp_mean, temp_variance = kf.predict(temp_mean, temp_variance)
            print("Estimated position in frame ", frame_num, ": ", temp_mean[:2])

            if isMultipleObjectsDetected:
                ball_in_play = get_ball_in_play(measurement, ball_in_play)
            else:
                ball_in_play = measurement[0]

            # Eliminate false positives by using threshold distance between ball in frame t and t-1
            alpha = 0.5 + (len(missed_frames) / MAX_NUM_CONSECUTIVE_MISS * 2)
            if not que_future.empty():
                cmp_frame = que_future.queue[-1][1]
                dist = euclidean_distance(que_future.queue[-1][0][:2], xyxy_to_xyah(ball_in_play)[:2])  
                dist_threshold = euclidean_distance(temp_mean[:2], que_future.queue[-1][0][:2])  
                print("Distance between ball in frame ", que_future.queue[-1][1], que_future.queue[-1][0][:2], "and ", frame_num, xyxy_to_xyah(ball_in_play)[:2], ": ", dist)
            else:
                cmp_frame = que_past.queue[-1][1]
                dist = euclidean_distance(que_past.queue[-1][0][:2], xyxy_to_xyah(ball_in_play)[:2])  
                dist_threshold = euclidean_distance(temp_mean[:2], que_past.queue[-1][0][:2])  
                print("Distance between ball in frame ", que_past.queue[-1][1], que_past.queue[-1][0][:2], "and ", frame_num, xyxy_to_xyah(ball_in_play)[:2], ": ", dist)
            
            distance_threshold = (alpha * (MAX_DISTANCE_PER_FRAME + 20 * (frame_num - cmp_frame)) + (1 - alpha) * (dist_threshold + 50))
            print("Distance estimation from kalman: ", dist_threshold + 50)
            print("Acceleration xa, ya: ", temp_mean[8:10])
            print("Distance heuristic: ", MAX_DISTANCE_PER_FRAME + 20 * (frame_num - cmp_frame))
            print("Distance Threshold: ", distance_threshold)
            if dist > distance_threshold: 
                isObjectDetected = False
                print("***False Positive Detected***")
            
            
        if not isTrackInitialised:
            if isObjectDetected:
                ball_in_play = measurement[0]
                
                # TODO: After reset during reinitializing tracker, implement to filter out false positive
                if isAfterReset:
                    if isMultipleObjectsDetected:
                        ball_in_play = get_ball_in_play(measurement, ball_in_play)
                        # ball_in_play = get_ball_in_play(measurement, xyah_to_xyxy(temp_mean[0:4]))
                    else:
                        ball_in_play = measurement[0]

                    if not que_future.empty():
                        cmp_frame = que_future.queue[-1][1]
                        dist = euclidean_distance(que_future.queue[-1][0][:2], xyxy_to_xyah(ball_in_play)[:2])  
                        dist_threshold = euclidean_distance(temp_mean[:2], que_future.queue[-1][0][:2])  
                        print("Distance between ball in frame ", que_future.queue[-1][1], que_future.queue[-1][0][:2], "and ", frame_num, xyxy_to_xyah(ball_in_play)[:2], ": ", dist)
                    else:
                        cmp_frame = que_past.queue[-1][1]
                        dist = euclidean_distance(que_past.queue[-1][0][:2], xyxy_to_xyah(ball_in_play)[:2])  
                        dist_threshold = euclidean_distance(temp_mean[:2], que_past.queue[-1][0][:2])  
                        print("Distance between ball in frame ", que_past.queue[-1][1], que_past.queue[-1][0][:2], "and ", frame_num, xyxy_to_xyah(ball_in_play)[:2], ": ", dist)

                    distance_threshold = (MAX_DISTANCE_PER_FRAME + 20 * (frame_num - cmp_frame) + dist_threshold + 20)/2
                    print("Distance Threshold: ", (MAX_DISTANCE_PER_FRAME + 20 * (frame_num - cmp_frame) + dist_threshold + 20)/2)

                    if dist > distance_threshold:
                        print("***Tracking re-initialization failed***")
                        print("***False Positive Detected***")
                        # cv2.imwrite(f"missed_frames/{filename}_{frame_num}.jpg", frame)
                        cv2.putText(frame, "Frame: " + str(frame_num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)
                        vidwriter.write(frame)
                    else:
                        que_past.queue.clear()
                        que_future.queue.clear()
                        isAfterReset = False
                        print("***New Tracking Initiated***")
                        mean, variance = kf.initiate(xyxy_to_xyah(ball_in_play))
                        temp_mean = mean
                        temp_variance = variance
                        enqueue(que=que_past, data=(xyxy_to_xyah(ball_in_play), frame_num))
                        isTrackInitialised = True
                        annotated_frame = annotate(frame, ball_in_play, color=(0, 0, 255), method="detect", thickness=2, frame_num=frame_num)
                        # annotated_frame = annotate(annotated_frame, xyah_to_xyxy(mean[0:4]), color=(0, 0, 0), method="kalman")
                        annotated_frame = draw_trails(annotated_frame, ball_in_play, que_trailmarks, color=(0, 0, 255))
                        vidwriter.write(annotated_frame)
                else:
                    que_past.queue.clear()
                    que_future.queue.clear()
                    mean, variance = kf.initiate(xyxy_to_xyah(ball_in_play))
                    temp_mean = mean
                    temp_variance = variance
                    enqueue(que=que_past, data=(xyxy_to_xyah(ball_in_play), frame_num))
                    isTrackInitialised = True
                    annotated_frame = annotate(frame, ball_in_play, color=(0, 0, 255), method="detect", thickness=2, frame_num=frame_num)
                    # annotated_frame = annotate(annotated_frame, xyah_to_xyxy(mean[0:4]), color=(0, 0, 0), method="kalman")
                    annotated_frame = draw_trails(annotated_frame, ball_in_play, que_trailmarks, color=(0, 0, 255))
                    vidwriter.write(annotated_frame)

            else:
                # cv2.imwrite(f"missed_frames/{filename}_{frame_num}.jpg", frame)
                cv2.putText(frame, "Frame: " + str(frame_num), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)
                vidwriter.write(frame)
        
        else: 
            if isObjectDetected:
                if isBufferOn:
                    if que_future.qsize() == que_future.maxsize - 1:
                        que_future.put((xyxy_to_xyah(ball_in_play), frame_num, frame, "future"))

                        x_model, y_model, a_model, h_model = get_xyah_interpolation_model(que_past, que_future)

                        interpolated_missed_frames = interpolate_missed_frames(x_model, y_model, a_model, h_model, missed_frames)

                        held_frames = sorted(interpolated_missed_frames + list(que_future.queue), key=lambda x: x[1])

                        mean, variance = process_buffered_frames(mean, variance, held_frames, que_future, que_trailmarks, vidwriter)
                        temp_mean = mean
                        temp_variance = variance

                        isBufferOn = False
                        missed_frames = []
                        held_frames = []
                        x, y, a, h, t = [], [], [], [], []
                        interpolated_missed_frames = []
                        que_future.queue.clear()
                    else:
                        que_future.put((xyxy_to_xyah(ball_in_play), frame_num, frame, "future"))
                else:
                    enqueue(que=que_past, data=(xyxy_to_xyah(ball_in_play), frame_num))
                    mean, variance = kf.predict(mean, variance)
                    mean , variance = kf.update(mean, variance, xyxy_to_xyah(ball_in_play))
                    print("Current state: ", np.round(mean[:4], 2))
                    temp_mean = mean
                    temp_variance = variance

                    annotated_frame = annotate(frame, ball_in_play, color=(0, 0, 255), method="detect", thickness=2, frame_num=frame_num)
                    # annotated_frame = annotate(annotated_frame, xyah_to_xyxy(mean[0:4]), color=(0, 0, 0), method="kalman")
                    annotated_frame = draw_trails(annotated_frame,ball_in_play, que_trailmarks)
                    vidwriter.write(annotated_frame)
            else:
                missed_frames.append((frame, frame_num))
                temp_mean, temp_variance = kf.predict(temp_mean, temp_variance)

                isBufferOn = True

                # !! TODO: Bug fix Even missed frames exceed the limit, extrapolation is still performed
                if len(missed_frames) >= MAX_NUM_CONSECUTIVE_MISS:
                    if not que_future.empty():
                        x_model, y_model, a_model, h_model = get_xyah_interpolation_model(que_past, que_future)
                       
                        interpolated_missed_frames = interpolate_missed_frames(x_model, y_model, a_model, h_model, missed_frames)

                        held_frames = sorted(interpolated_missed_frames + list(que_future.queue), key=lambda x: x[1])
                
                        mean, variance = process_buffered_frames(mean, variance, held_frames, que_future, que_trailmarks, vidwriter)
                        temp_mean = mean
                        temp_variance = variance
                    else:
                        first_missed_frame_num = missed_frames[0][1]
                        for i, (missed_frame, frame_n) in enumerate(missed_frames):
                            annotated_frame = missed_frame.copy()
                            if i < 3:
                                mean, variance = kf.predict(mean, variance)
                                annotated_frame = annotate(missed_frame, xyah_to_xyxy(mean[0:4]), color=(255, 0, 0), method="kalman", frame_num=frame_n)
                                annotated_frame = draw_trails(annotated_frame, xyah_to_xyxy(mean[0:4]), que_trailmarks, color=(255, 0, 0))
                            else:
                                # cv2.imwrite(f"missed_frames/{filename}_{frame_n}.jpg", annotated_frame)
                                cv2.putText(annotated_frame, "Frame: " + str(frame_n), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)
                            vidwriter.write(annotated_frame)


                    # Reset everything
                    isTrackInitialised = False
                    isObjectedDetected = False
                    isBufferOn = False
                    isMultipleObjectsDetected = False
                    isAfterReset = True

                    missed_frames = []
                    interpolated_missed_frames = []
                    x, y, a, h, t = [], [], [], [], []

                    # que_future.queue.clear()
                    # que_past.queue.clear()
                    que_trailmarks.queue.clear()
        
        if frame_num == total_frames -1 :
            if len(missed_frames) > 0:
                if not que_future.empty():
                    x_model, y_model, a_model, h_model = get_xyah_interpolation_model(que_past, que_future)
                    
                    interpolated_missed_frames = interpolate_missed_frames(x_model, y_model, a_model, h_model, missed_frames)

                    held_frames = sorted(interpolated_missed_frames + list(que_future.queue), key=lambda x: x[1])
                    mean, variance = process_buffered_frames(mean, variance, held_frames, que_future, que_trailmarks, vidwriter)
                else:
                    for i, (missed_frame, frame_n) in enumerate(missed_frames):
                        annotated_frame = missed_frame.copy()
                        # cv2.imwrite(f"missed_frames/{filename}_{frame_n}.jpg", annotated_frame)
                        cv2.putText(annotated_frame, "Frame: " + str(frame_n), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)
                        vidwriter.write(annotated_frame)
                                        
    vidwriter.release()
