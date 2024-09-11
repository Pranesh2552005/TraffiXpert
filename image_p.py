import cv2
import numpy as np
import time

signal1 = "(9.921, 78.116)"
signal2 = "(9.922, 78.120)"

# Paths to YOLOv3 files
weights_path = r"yolov3.weights"
cfg_path = r"yolov3.cfg"
coco_names_path = r"coco.names.txt"

# Load YOLOv3
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    car_count = 0
    truck_count = 0
    
    for out in outs:
        for detection in out:
            if len(detection) >= 85:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3 and classes[class_id] in ['car', 'truck']:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            if label == 'car':
                car_count += 1
            elif label == 'truck':
                truck_count += 1
    
    return car_count, truck_count

def process_video(video_path, duration):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0, 0

    total_car_count = 0
    total_truck_count = 0
    frame_count = 0

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        car_count, truck_count = detect_objects(frame)
        total_car_count += car_count
        total_truck_count += truck_count
        frame_count += 1

        elapsed_time = time.time() - start_time
        if elapsed_time > duration:
            break

    cap.release()

    avg_car_count = total_car_count // frame_count if frame_count > 0 else 0
    avg_truck_count = total_truck_count // frame_count if frame_count > 0 else 0

    return avg_car_count, avg_truck_count

# Paths to the input videos (replace these paths with valid video file paths)
video_path1 = "2103099-uhd_3840_2160_30fps.mp4"
video_path2 = "19696722-hd_1080_1920_30fps.mp4"

# Process each video and calculate averages
def calculate_duration(video_path, duration, rate=1.5):
    avg_car_count, avg_truck_count = process_video(video_path, duration)
    combined_count = avg_car_count + avg_truck_count

    duration_per_vehicle = rate * combined_count if combined_count > 0 else float('inf')
    return min(duration_per_vehicle, 120), avg_car_count, avg_truck_count

def signal_control():
    duration = 3  # Duration to process videos in seconds
    signal_duration1, avg_car_count1, avg_truck_count1 = calculate_duration(video_path1, duration)
    signal_duration2, avg_car_count2, avg_truck_count2 = calculate_duration(video_path2, duration)
    
    while True:
        # Signal 1 is high
        print(f"\033[92mSignal 1 (Video 1) is high for {signal_duration1:.2f} seconds\033[0m")
        start_time = time.time()

        # Run Signal 1 for calculated duration
        while time.time() - start_time < signal_duration1 - 5:  # Switch in last 5 seconds
            time.sleep(1)  # Placeholder for signal active state

        # Calculate Signal 2 time during the last 5 seconds of Signal 1
        signal_duration2, _, _ = calculate_duration(video_path2, duration)

        print(f"\033[93mSwitching to Signal 2 in 5 seconds...\033[0m")
        time.sleep(5)

        # Signal 2 is high
        print(f"\033[92mSignal 2 (Video 2) is high for {signal_duration2:.2f} seconds\033[0m")
        start_time = time.time()

        # Run Signal 2 for calculated duration
        while time.time() - start_time < signal_duration2 - 5:
            time.sleep(1)  # Placeholder for signal active state

        # Calculate Signal 1 time during the last 5 seconds of Signal 2
        signal_duration1, _, _ = calculate_duration(video_path1, duration)

        print(f"\033[93mSwitching to Signal 1 in 5 seconds...\033[0m")
        time.sleep(5)

# Start the signal control loop
signal_control()
