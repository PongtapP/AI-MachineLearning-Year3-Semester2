import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse
import discord
import asyncio
import logging
import threading
import os

# ตั้งค่าการบันทึก log
logging.basicConfig(
    level=logging.INFO,  # ระดับการบันทึก log (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # รูปแบบของ log
    filename='fire_discord.log',  # บันทึก log ลงไฟล์
    filemode='w'  # 'w' สำหรับเขียนทับไฟล์เดิม, 'a' สำหรับเพิ่ม log ต่อท้าย
)

# Discord settings
DISCORD_TOKEN = 'MTM0ODgyNDQyMTQ4OTU3MzkwOQ.GT0ad8.chdltb9kPJg1hXOWebqPGt83UhbAAjM76vJxPE'
DISCORD_CHANNEL_ID = 1348828995558441010  # แทนที่ด้วย channel id ของคุณ

# Global variables for cooldown
last_alert_time = 0
COOLDOWN = 300  # 5 minutes in seconds

# Initialize Discord client
intents = discord.Intents.default()
client = discord.Client(intents=intents)

def send_discord_alert(message, image_path=None):
    global last_alert_time
    current_time = time.time()

    if current_time - last_alert_time >= COOLDOWN:
        try:
            logging.info("Attempting to send Discord alert...")
            print("Attempting to send Discord alert...")
            channel = client.get_channel(DISCORD_CHANNEL_ID)
            if channel:
                if image_path:
                    # ส่งภาพพร้อมข้อความ
                    with open(image_path, 'rb') as f:
                        picture = discord.File(f)
                        asyncio.run_coroutine_threadsafe(channel.send(message, file=picture), client.loop)
                    # ลบไฟล์ภาพหลังจากส่งเสร็จ
                    os.remove(image_path)
                    logging.info(f"Image deleted: {image_path}")
                else:
                    # ส่งเฉพาะข้อความ
                    asyncio.run_coroutine_threadsafe(channel.send(message), client.loop)
                last_alert_time = current_time
                logging.info(f"Alert sent to Discord: {message}")
                print(f"Alert sent to Discord: {message}")
            else:
                logging.error("Error: Discord channel not found")
                print("Error: Discord channel not found")
        except Exception as e:
            logging.error(f"Error sending Discord alert: {e}")
            print(f"Error sending Discord alert: {e}")
            # ลบไฟล์ภาพหากเกิดข้อผิดพลาด
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                logging.info(f"Image deleted due to error: {image_path}")
    else:
        # ลบไฟล์ภาพหากอยู่ในช่วง cooldown
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            logging.info(f"Image deleted due to cooldown: {image_path}")

def parse_args():
    """Parse command line arguments."""
    logging.info("Parsing command line arguments...")
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Fire and Smoke Detection using YOLOv8")
    parser.add_argument('--model', type=str, default='best8.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    return parser.parse_args()

def detection_loop(model, cap, args):
    """Function to handle the detection loop."""
    # Define class names and colors
    class_names = {0: 'fire', 1: 'smoke'}
    colors = {0: (0, 0, 255), 1: (128, 128, 128)}

    # Variables for FPS calculation
    frame_count = 0
    fps_avg = 0
    fps_start_time = time.time()

    # Detection interval
    detection_interval = 2  # Detect every 2 seconds
    last_detection_time = 0

    logging.info("Starting realtime detection.")
    print("Starting realtime detection.")
    while True:
        # Read a frame from the webcam
        logging.debug("Reading frame from webcam...")
        try:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to receive frame from webcam")
                print("Failed to receive frame from webcam")
                break
        except Exception as e:
            logging.error(f"Error reading frame from webcam: {e}")
            print(f"Error reading frame from webcam: {e}")
            break

        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            last_detection_time = current_time

            # Perform detection
            logging.debug("Performing detection...")
            try:
                results = model(frame, conf=args.conf, verbose=False)[0]
                detections = results.boxes.data.cpu().numpy()
                logging.debug("Detection completed.")
            except Exception as e:
                logging.error(f"Error during detection: {e}")
                continue

            # Check if fire or smoke detected
            fire_detected = False
            smoke_detected = False

            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                class_id = int(class_id)

                class_name = class_names.get(class_id, f"Class {class_id}")
                color = colors.get(class_id, (0, 255, 0))

                if class_id == 0:  # fire
                    fire_detected = True
                elif class_id == 1:  # smoke
                    smoke_detected = True

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if fire_detected or smoke_detected:
                # บันทึกภาพเมื่อตรวจจับไฟหรือควัน
                image_path = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(image_path, frame)
                logging.info(f"Image saved: {image_path}")

                # ส่งภาพไปยัง Discord
                if fire_detected:
                    message = "YOLO8n: Fire detected!"
                else:
                    message = "YOLO8n: Smoke detected!"
                threading.Thread(target=send_discord_alert, args=(message, image_path)).start()

        # Show the result
        cv2.imshow("Fire and Smoke Detection (Realtime)", frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("User pressed 'q' to quit")
            print("User pressed 'q' to quit")
            break

    # Release resources
    logging.info("Releasing resources...")
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Detection stopped")
    print("Detection stopped")

async def discord_loop():
    """Function to handle the Discord connection."""
    logging.info("Connecting to Discord...")
    print("Connecting to Discord...")
    try:
        await client.start(DISCORD_TOKEN)
        logging.info("Connected to Discord successfully.")
        print("Connected to Discord successfully.")
    except Exception as e:
        logging.error(f"Error connecting to Discord: {e}")
        print(f"Error connecting to Discord: {e}")
        return

async def main():
    # Parse arguments
    args = parse_args()
    logging.info("Starting fire-discord.py with arguments: %s", args)
    print("Starting fire-discord.py with arguments:", args)

    # Step 1: Load the YOLOv8 model
    logging.info(f"Loading model {args.model}...")
    print(f"Loading model {args.model}...")
    try:
        model = YOLO(args.model)
        logging.info("Model loaded successfully.")
        print("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return

    # Step 2: Open the webcam
    logging.info(f"Opening webcam at index {args.camera}...")
    print(f"Opening webcam at index {args.camera}...")
    try:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            logging.error(f"Error: Could not open webcam at index {args.camera}")
            print(f"Error: Could not open webcam at index {args.camera}")
            return
        else:
            logging.info("Webcam opened successfully.")
            print("Webcam opened successfully.")
    except Exception as e:
        logging.error(f"Error opening webcam: {e}")
        print(f"Error opening webcam: {e}")
        return

    # Step 3: Start detection loop in a separate thread
    detection_thread = threading.Thread(target=detection_loop, args=(model, cap, args))
    detection_thread.start()

    # Step 4: Start Discord loop
    await discord_loop()

    # Wait for the detection thread to finish
    detection_thread.join()

if __name__ == "__main__":
    try:
        logging.info("Starting fire-discord.py...")
        print("Starting fire-discord.py...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Stopping detection...")
        print("Stopping detection...")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")