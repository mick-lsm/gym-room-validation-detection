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
import threading
import asyncio

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
            winsound.Beep(1000, 2000)  # Frequency: 1000Hz, Duration: 200ms
            time.sleep(0.5)  # Short pause between beeps
        time.sleep(1.2)


def warn(caps, model, delayedSeconds):
    print("Too many people")
    threading.Thread(target=beep_beep_alert, daemon=True).start()
    threading.Thread(
        target=send_email,
        args=(
            os.getenv("sender"),
            os.getenv("password"),
            os.getenv("receiver"),
            "Gym Room Alert",
            f"Please check the gym room. The images of the cameras are attached and the detection has been delayed for {delayedSeconds} seconds.",
            caps,
            model
        ),
        daemon=True
    ).start()

def send_email(sender_username, sender_password, recipient_email, subject, message_body, caps, model):
    """
    Send an email with the current frame of the video capture array as an attachment,
    drawing human detection boxes on the frame.
    """
    try:
        # Set up the MIME
        message = MIMEMultipart()
        message['From'] = sender_username
        message['To'] = recipient_email
        message['Subject'] = subject

        # Attach the message body
        message.attach(MIMEText(message_body, 'plain'))

        for i, cap in enumerate(caps):
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Run YOLO detection
                    results = model(frame, verbose=False)
                    # Draw detection boxes on the frame
                    annotated_frame = results[0].plot()
                    cv2.putText(annotated_frame, f'People Count: {len(results[0].boxes)}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # No color conversion needed, keep BGR for PNG/JPEG
                    _, img_encoded = cv2.imencode('.png', annotated_frame)
                    ext = "png"
                    

                    img_bytes = img_encoded.tobytes()

                    # Create and attach image
                    img_attachment = MIMEImage(img_bytes, name=f"camera_{i+1}.{ext}")
                    message.attach(img_attachment)
                else:
                    print(f"⚠️ Failed to capture frame from camera {i+1}")
            else:
                print(f"⚠️ Camera {i+1} is not accessible")

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

        if(current_time - start_time >= int(os.getenv("delay"))): 
            detected = False
        if(detected == True):
            continue


        average = calculate_people_count_by_frame(model, caps)
        
        if(average > max_in_5_secs):
            max_in_5_secs = average
        

        if(current_time - start_time > 5):
            print(f'current time : {current_time}, people_count : {average}')
            if(max_in_5_secs > limit):
                warn(caps, model, int(os.getenv("delay")))
                detected = True

            start_time = current_time
            max_in_5_secs = 0


if __name__ == "__main__":
    async def async_main():
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(None, YOLO, os.getenv("model"))
        caps = [
            await loop.run_in_executor(None, cv2.VideoCapture, int(os.getenv("cam1"))),
            await loop.run_in_executor(None, cv2.VideoCapture, int(os.getenv("cam2")))
        ]
        await loop.run_in_executor(None, main, int(os.getenv("limit")), model, caps)

    asyncio.run(async_main())