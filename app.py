import cv2
from ultralytics import YOLO
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import os
import winsound

load_dotenv()  # Loads variables from .env into the environment

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


def beep_beep_alert():
    for __ in range(3):  # Nine beeps
        for _ in range(3):  # Three beeps
            winsound.Beep(1000, 200)  # Frequency: 1000Hz, Duration: 200ms
            time.sleep(0.1)  # Short pause between beeps


def warn():
    print("Too many people")
    beep_beep_alert()
    send_email(os.getenv("sender"), os.getenv("password"), os.getenv("receiver"), "Too many people", "Too many people")

def send_email(sender_username, sender_password, recipient_email, subject, message_body, caps):
    """
    Send an email with the current frame of the video capture array as an attachment.
    
    Parameters:
    - sender_username: Your email address (e.g., 'your_email@gmail.com')
    - sender_password: Your email password or app-specific password
    - recipient_email: The recipient's email address
    - subject: Email subject line
    - message_body: The content of the email
    - caps: Video capture array (OpenCV frame) to attach as an image
    
    Returns:
    - True if email sent successfully, False otherwise
    """
    try:
        # Set up the MIME
        message = MIMEMultipart()
        message['From'] = sender_username
        message['To'] = recipient_email
        message['Subject'] = subject
        
        # Attach the message body
        message.attach(MIMEText(message_body, 'plain'))
        
        # Save the current frame as an image and attach it
        if caps is not None:
            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(caps, cv2.COLOR_BGR2RGB)
            
            # Encode the frame as JPEG in memory
            _, img_encoded = cv2.imencode('.jpg', frame_rgb)
            img_bytes = img_encoded.tobytes()
            
            # Attach the image
            img_attachment = MIMEImage(img_bytes, name='current_frame.jpg')
            message.attach(img_attachment)
        
        # SMTP server configuration (for Gmail)
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587
        
        # Create SMTP session
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Enable security
            server.login(sender_username, sender_password)  # Login
            text = message.as_string()
            server.sendmail(sender_username, recipient_email, text)
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
    

def main(limit, model, caps):
    detected = False

    max_in_5_secs = 0

    start_time = time.time()
    while(True):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        current_time = time.time()

        if(current_time - start_time >= os.getenv("duration")): 
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
    main(int(os.getenv("limit")), YOLO(os.getenv("model")), [cv2.VideoCapture(int(os.getenv("cam1"))), cv2.VideoCapture(int(os.getenv("cam2")))])