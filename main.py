import argparse
from ultralytics import YOLO
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Global variables
last_frame_boxes = [[0, 0, 0, 0, 0, 0]]
all_frames = []
abounded_luggage = []

def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def compare_boxes(boxes_frame_n, boxes_frame_np1, distance_threshold=0.2):
    stationary_objects = []
    for idx_n, (x1_n, y1_n, x2_n, y2_n, _, _) in enumerate(boxes_frame_n):
        center_n = get_center(x1_n, y1_n, x2_n, y2_n)
        for idx_np1, (x1_np1, y1_np1, x2_np1, y2_np1, _, _) in enumerate(boxes_frame_np1):
            center_np1 = get_center(x1_np1, y1_np1, x2_np1, y2_np1)
            distance = calculate_distance(center_n[0], center_n[1], center_np1[0], center_np1[1])
            if distance <= distance_threshold:
                return True
                stationary_objects.append(distance)
                break
    return False

def compare_luggage(boxes_frame_n, boxes_frame_np1, distance_threshold=30):
    for idx_n, (x1_n, y1_n, x2_n, y2_n, _, _) in enumerate(boxes_frame_n):
        center_n = get_center(x1_n, y1_n, x2_n, y2_n)
        for idx_np1, (x1_np1, y1_np1, x2_np1, y2_np1, _, _) in enumerate(boxes_frame_np1):
            center_np1 = get_center(x1_np1, y1_np1, x2_np1, y2_np1)
            distance = calculate_distance(center_n[0], center_n[1], center_np1[0], center_np1[1])
            if distance > distance_threshold:
                return True
            else:
                return False

def find_closest_box(box0, class_0_boxes):
    center0 = (int((box0[0] + box0[2]) / 2), int((box0[1] + box0[3]) / 2))
    min_distance = float('inf')
    closest_box = None
    closest_center = None
    
    for box in class_0_boxes:
        center = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        distance = calculate_distance(center0[0], center0[1], center[0], center[1])
        if distance < min_distance:
            min_distance = distance
            closest_box = box
            closest_center = center
            
    return (box0, closest_box, min_distance, center0, closest_center)

def draw_box(the_box, the_frame, abound, color):    
    x1, y1, x2, y2, score, classes = the_box
    if abound:
        class_name = "Abandoned luggage"
    else:
        class_name = model.names[classes]       
    cv2.rectangle(the_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.putText(the_frame, f"class: {class_name}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(the_frame, f"conf: {round(score,2)}", (int(x2), int(y2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def check_proximity(boxes, frame, threshold, frame_count):
    global last_frame_boxes, all_frames, abounded_luggage
    
    class_0_boxes = [box for box in boxes if box[5] == 0]
    other_classes_boxes = [box for box in boxes if box[5] in [24, 26, 28]]  
    all_frames.append(other_classes_boxes)
    
    if frame_count >= 60:
        last_frame_boxes = all_frames[frame_count - 30]
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda box0: find_closest_box(box0, class_0_boxes), other_classes_boxes)
    
    for result in results:
        box0, closest_box, distance, center0, closest_center = result
        if closest_box is None:
            draw_box(box0, frame, True, (0, 255, 255))
            continue
        if closest_box and distance <= threshold:
            draw_box(closest_box, frame, False, (140, 50, 60))
            draw_box(box0, frame, False, (0, 255, 0))
            cv2.line(frame, center0, closest_center, (255, 0, 0), thickness=3)
        elif distance > threshold:
            if compare_boxes([box0], last_frame_boxes):
                if len(abounded_luggage) == 0:
                    abounded_luggage.append(box0)
                    draw_box(closest_box, frame, False, (0, 0, 255))
                    draw_box(box0, frame, False, (0, 255, 255))
                    cv2.line(frame, center0, closest_center, (255, 0, 0), thickness=3)  
                    cv2.imwrite("saves/" + str(frame_count) + ".jpg", frame)
                if compare_luggage([box0], abounded_luggage):
                    draw_box(closest_box, frame, False, (0, 0, 255))
                    draw_box(box0, frame, False, (0, 255, 255))
                    cv2.line(frame, center0, closest_center, (255, 0, 0), thickness=3)  
                    cv2.imwrite("saves/" + str(frame_count) + ".jpg", frame)
                    abounded_luggage.append(box0)
                else:
                    draw_box(box0, frame, True, (0, 255, 255))

    return frame

def main(video_path, model_name, device, confidence, threshold):
    global model
    model = YOLO(model_name)
    model.to(device)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, main_frame = cap.read()
        if not ret:
            break 
        
        results = model(main_frame, classes=[0, 24, 26, 28], device=device, conf=confidence, verbose=False)
        
        processed_frame = check_proximity(results[0].boxes.data.tolist(), main_frame, threshold, frame_count)
        cv2.imshow('YOLO', processed_frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Video Processing")
    parser.add_argument("--video", type=str, default="data/Dataset_Videos/Video1_DS.mp4", help="Path to the video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model name or path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--confidence", type=float, default=0.6, help="Detection confidence threshold")
    parser.add_argument("--threshold", type=int, default=100, help="Proximity threshold")
    
    args = parser.parse_args()
    
    main(args.video, args.model, args.device, args.confidence, args.threshold)
