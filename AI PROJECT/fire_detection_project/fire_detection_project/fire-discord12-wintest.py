import cv2
import time
from ultralytics import YOLO
import argparse
import discord
import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os

# ตั้งค่าระบบ Logging
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    file_handler = RotatingFileHandler('logs/fire_detection.log', maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    error_handler = RotatingFileHandler('logs/error.log', maxBytes=2*1024*1024, backupCount=1, encoding='utf-8')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(file_formatter)
    
    main_logger.addHandler(console_handler)
    main_logger.addHandler(file_handler)
    main_logger.addHandler(error_handler)
    
    logging.getLogger('discord').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('ultralytics').setLevel(logging.WARNING)

# ตั้งค่า Discord (แก้ไข Token และ Channel ID ตามของจริง)
DISCORD_TOKEN = 'MTM0ODgyNDQyMTQ4OTU3MzkwOQ.GT0ad8.chdltb9kPJg1hXOWebqPGt83UhbAAjM76vJxPE'
DISCORD_CHANNEL_ID = 1354418108126855171

class FireDetectionSystem:
    def __init__(self):
        self.exit_flag = False
        self.last_alert_time = 0
        self.cooldown = 300  # 5 นาที (หน่วย: วินาที)
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize_discord(self):
        """เริ่มต้น Discord Client"""
        intents = discord.Intents.default()
        intents.message_content = True  # จำเป็นสำหรับส่งข้อความ
        self.client = discord.Client(intents=intents)
        
        @self.client.event
        async def on_ready():
            self.logger.info(f'✅ Bot พร้อมทำงาน: {self.client.user}')
        
        # เริ่มเชื่อมต่อ Discord แบบ async
        await self.client.start(DISCORD_TOKEN)
    
    async def run_detection(self, camera_index, model_path):
        """ระบบตรวจจับภาพหลัก"""
        try:
            # ตรวจสอบการเปิดกล้อง
            self.logger.info(f"🔍 กำลังเปิดกล้อง index: {camera_index}")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                self.logger.error("❌ เปิดกล้องล้มเหลว!")
                return
            
            # โหลดโมเดล YOLO (รองรับ YOLOv8 และ YOLO12x)
            self.logger.info(f"🔄 กำลังโหลดโมเดล: {model_path}")
            model = YOLO(model_path)  # ใช้ YOLO12x ได้หากติดตั้งถูกต้อง
            self.logger.info("✅ โหลดโมเดลสำเร็จ")
            
            # ลูปตรวจจับภาพ
            self.logger.info("🚀 เริ่มการตรวจจับ...")
            while not self.exit_flag:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("❌ รับภาพจากกล้องล้มเหลว")
                    break
                
                # ประมวลผลภาพ
                results = model(frame, conf=0.25, verbose=False)
                self.process_results(frame, results[0])
                
                # แสดงผลหน้าต่าง OpenCV
                cv2.imshow('Fire Detection', results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.exit_flag = True
                    
        except Exception as e:
            self.logger.error(f"⚠️ ข้อผิดพลาดการตรวจจับ: {str(e)}", exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("🛑 ปิดระบบตรวจจับ")

    def process_results(self, frame, results):
        if len(results.boxes) == 0:
            return
        for box in results.boxes:
            class_name = results.names[int(box.cls)]
            if class_name in ['fire', 'smoke']:
                self.handle_detection(frame, class_name)

    def handle_detection(self, frame, class_name):
        current_time = time.time()
        if current_time - self.last_alert_time < self.cooldown:
            self.logger.debug(f"⏳ อยู่ในช่วง cooldown ({class_name})")
            return
        
        # บันทึกรูปและส่งการแจ้งเตือน
        image_path = f"detection_{int(time.time())}.jpg"
        cv2.imwrite(image_path, frame)
        self.logger.info(f"📸 บันทึกรูป: {image_path}")

        message = "🔥 ตรวจพบไฟ!" if class_name == 'fire' else "💨 ตรวจพบควัน!"
        asyncio.run_coroutine_threadsafe(self.send_discord_alert(message, image_path), self.client.loop)
        self.last_alert_time = current_time

    async def send_discord_alert(self, message, image_path):
        try:
            if self.client.is_closed():
                self.logger.warning("❌ Discord Client ปิดแล้ว")
                return
            
            channel = self.client.get_channel(DISCORD_CHANNEL_ID)
            if channel is None:
                self.logger.error(f"❌ ไม่พบช่อง Discord (ID: {DISCORD_CHANNEL_ID})")
                return
            
            with open(image_path, 'rb') as f:
                await channel.send(message, file=discord.File(f))
            
            self.logger.info("📤 ส่งการแจ้งเตือนสำเร็จ")
        except Exception as e:
            self.logger.error(f"⚠️ ส่งการแจ้งเตือนล้มเหลว: {str(e)}", exc_info=True)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
                self.logger.debug(f"🗑️ ลบไฟล์: {image_path}")

async def main(args):
    setup_logging()
    logger = logging.getLogger(__name__)
    system = FireDetectionSystem()
    
    try:
        # เริ่ม Discord และตรวจจับพร้อมกัน
        await asyncio.gather(
            system.initialize_discord(),
            system.run_detection(args.camera, args.model)
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 หยุดโดยผู้ใช้")
    except Exception as e:
        logger.error(f"⚠️ ข้อผิดพลาดหลัก: {str(e)}", exc_info=True)
    finally:
        system.exit_flag = True
        if system.client and not system.client.is_closed():
            await system.client.close()
        logger.info("✅ ระบบปิดทำงานเสร็จสิ้น")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Index กล้อง (default: 0)')
    parser.add_argument('--model', type=str, required=True, help='พาธไฟล์โมเดล YOLO (.pt)')
    args = parser.parse_args()
    
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nผู้ใช้สั่งหยุดการทำงาน")