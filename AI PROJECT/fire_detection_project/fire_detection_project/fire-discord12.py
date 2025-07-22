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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='fire_discord.log',
    filemode='w'
)

# Discord settings
DISCORD_TOKEN = 'MTM0ODgyNDQyMTQ4OTU3MzkwOQ.GT0ad8.chdltb9kPJg1hXOWebqPGt83UhbAAjM76vJxPE'
DISCORD_CHANNEL_ID = 1348828995558441010

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
                    with open(image_path, 'rb') as f:
                        picture = discord.File(f)
                        asyncio.run_coroutine_threadsafe(channel.send(message, file=picture), client.loop)
                    os.remove(image_path)
                    logging.info(f"Image deleted: {image_path}")
                else:
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
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                logging.info(f"Image deleted due to error: {image_path}")

def parse_args():
    """Parse command line arguments."""
    logging.info("Parsing command line arguments...")
    print("Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Fire and Smoke Detection using YOLO12x")
    parser.add_argument('--model', type=str, default='best12.pt', help='Path to YOLO12x model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    return parser.parse_args()

def detection_loop(model, cap, args):
    """Function to handle the detection loop."""
    class_names = {0: 'fire', 1: 'smoke'}
    colors = {0: (0, 0, 255), 1: (128, 128, 128)}

    frame_count = 0
    fps_avg = 0
    fps_start_time = time.time()

    detection_interval = 2
    last_detection_time = 0

    logging.info("Starting realtime detection with YOLO12x.")
    print("Starting realtime detection with YOLO12x.")
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to receive frame from webcam")
                break
        except Exception as e:
            logging.error(f"Error reading frame from webcam: {e}")
            break

        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            last_detection_time = current_time
            try:
                results = model(frame, conf=args.conf, verbose=False)[0]
                detections = results.boxes.data.cpu().numpy()
            except Exception as e:
                logging.error(f"Error during detection: {e}")
                continue

            fire_detected = any(int(detection[5]) == 0 for detection in detections)
            smoke_detected = any(int(detection[5]) == 1 for detection in detections)

            for detection in detections:
                x1, y1, x2, y2, conf, class_id = detection
                class_id = int(class_id)
                class_name = class_names.get(class_id, f"Class {class_id}")
                color = colors.get(class_id, (0, 255, 0))
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if fire_detected or smoke_detected:
                image_path = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(image_path, frame)
                message = "🔥 Fire detected using YOLO12x!" if fire_detected else "💨 Smoke detected using YOLO12x!"
                threading.Thread(target=send_discord_alert, args=(message, image_path)).start()

        cv2.imshow("Fire and Smoke Detection (YOLO12x)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def discord_loop():
    logging.info("Connecting to Discord...")
    try:
        await client.start(DISCORD_TOKEN)
    except Exception as e:
        logging.error(f"Error connecting to Discord: {e}")

async def main():
    args = parse_args()
    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.camera)
    detection_thread = threading.Thread(target=detection_loop, args=(model, cap, args))
    detection_thread.start()
    await discord_loop()
    detection_thread.join()

if __name__ == "__main__":
    asyncio.run(main())
