import cv2
import torch
import logging
from logging.handlers import RotatingFileHandler
import requests
from datetime import datetime
import argparse
import time
import os
from ultralytics import YOLO

# Configuration
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1354484813830160484/YQ2-spcfAFmqUAD_usbL9PqNwzCk5-2hYlALSnVXlyyCn_IHkJ-yDWTaGriAKNsyB73N'
COOLDOWN = 300  # 5 นาที (หน่วยวินาที)
CLASS_NAMES = ['Fire', 'Smoke']  # ตรวจสอบให้ตรงกับ class ในโมเดล
LOG_FILE = 'detection.log'

# สีกรอบและข้อความ (รูปแบบ BGR)
COLOR_CONFIG = {
    'Fire': {'bgr': (0, 255, 0), 'hex': 0x00FF00},  # สีเขียว
    'Smoke': {'bgr': (255, 0, 0), 'hex': 0x0000FF}   # สีน้ำเงิน
}

# Setup logging
logger = logging.getLogger('FireDetection')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=1024*1024*5,  # 5MB
    backupCount=3,
    encoding='utf-8'
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

last_alert_time = 0

def send_discord_alert(image_path, detections):
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time < COOLDOWN:
        logger.info(f"Alert suppressed (cooldown active)")
        return False
    
    try:
        # ตรวจสอบประเภทการตรวจจับ
        detected_types = set()
        for detection in detections:
            if detection.startswith('Fire'):
                detected_types.add('Fire')
            elif detection.startswith('Smoke'):
                detected_types.add('Smoke')
        
        # กำหนดประเภทและสี
        if 'Fire' in detected_types and 'Smoke' in detected_types:
            status = 'Both'
            embed_color = 0xFFA500  # สีส้ม
        elif 'Fire' in detected_types:
            status = 'Fire'
            embed_color = COLOR_CONFIG['Fire']['hex']
        elif 'Smoke' in detected_types:
            status = 'Smoke'
            embed_color = COLOR_CONFIG['Smoke']['hex']
        else:
            status = 'Unknown'
            embed_color = 0x808080  # สีเทา
        
        # สร้างข้อความแจ้งเตือน
        alert_messages = {
            'Fire': '🔥 Fire Detected!',
            'Smoke': '💨 Smoke Detected!',
            'Both': '🔥💨 Fire & Smoke Detected!'
        }
        
        message = {
            "content": f"⚠️ **{alert_messages.get(status, 'Unknown Alert')}** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "embeds": [{
                "title": "Detection Details",
                "description": "\n".join(detections),
                "color": embed_color,
                "image": {"url": "attachment://detection.jpg"}
            }]
        }
        
        with open(image_path, 'rb') as f:
            files = {'file': (image_path, f)}
            response = requests.post(DISCORD_WEBHOOK_URL, json=message, files=files)
            response.raise_for_status()
        
        last_alert_time = current_time
        logger.info(f"{status} alert sent to Discord")
        return True
    except Exception as e:
        logger.error(f"Failed to send Discord alert: {str(e)}")
        return False

def main(args):
    # โหลดโมเดล YOLO
    model = YOLO(args.model)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ตั้งค่าการทำนาย
    model.conf = 0.6  # Confidence threshold
    model.iou = 0.3   # NMS IoU threshold
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    logger.info(f"Starting fire detection system with {args.model}...")
    logger.info(f"Using {'CUDA' if torch.cuda.is_available() else 'CPU'} for inference")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Can't receive frame")
            break
        
        # ทำนาย
        results = model.predict(
            frame,
            imgsz=640,
            verbose=False,
            stream=False
        )
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = box.conf.item()
                
                if cls_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[cls_id]
                    label = f"{class_name} {conf:.2f}"
                    detections.append(label)
                    
                    # ตั้งค่าสีตามคลาส
                    color = COLOR_CONFIG[class_name]['bgr']
                    
                    # วาดกล่อง
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, 
                        (x1, y1),
                        (x2, y2),
                        color, 2
                    )
                    cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                    )
        
        # บันทึกและส่งการแจ้งเตือน
        if detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"detection_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            logger.info(f"Detection found: {', '.join(detections)}")
            
            # ส่งการแจ้งเตือน
            send_status = send_discord_alert(image_path, detections)
            
            # ลบไฟล์ไม่ว่าส่งสำเร็จหรือไม่
            try:
                os.remove(image_path)
                logger.info(f"Deleted image: {image_path}")
            except Exception as e:
                logger.error(f"Failed to delete image: {str(e)}")

        # แสดงผลลัพธ์
        cv2.imshow('Fire Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    logger.info("System shutdown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    args = parser.parse_args()
    
    main(args)