import cv2
from ultralytics import YOLO
import time

def calculate_people_count_by_frame(model, caps):
    average = 0

    for index, cap in enumerate(caps):
        if not cap.isOpened():
            print("Cannot open camera")
            return
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Run YOLOv8 pose estimation
        results = model(frame, verbose=False)

        # Draw keypoints and skeleton on the frame
        annotated_frame = results[0].plot()

        # Count number of people detected
        average *= index
        average += len(results[0].boxes)
        average /= index + 1

        cv2.putText(annotated_frame, f'People Count: {len(results[0].boxes)}', (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow(f'YOLOv8 Pose Detection {index}', annotated_frame)

    return round(average)

def warn():
    print("Too many people")

def main(limit, model, caps):
    detected = False

    max_in_5_secs = 0

    start_time = time.time()
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_time = time.time()

        if(current_time - start_time >= 120): 
            detected = False
        if(detected == True):
            continue


        average = calculate_people_count_by_frame(model, caps)
        
        if(average > max_in_5_secs):
            max_in_5_secs = average
        

        if(current_time - start_time > 5):
            print(f'current time : {current_time}, people_count : {average}')
            if(max_in_5_secs >= limit):
                warn()
                detected = True

            start_time = current_time
            max_in_5_secs = 0
        



if __name__ == "__main__":
    main(3, YOLO('yolov8m-pose.pt'), [cv2.VideoCapture(0), cv2.VideoCapture(2)])