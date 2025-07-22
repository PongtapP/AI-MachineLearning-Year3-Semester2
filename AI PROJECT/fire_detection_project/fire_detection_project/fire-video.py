# fire_detection_yolov12x_final.py
import os
import cv2
import torch
import argparse
from ultralytics import YOLO

class YOLOv12xDetector:
    def __init__(self, model_path, conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.model.conf = conf_thresh
        self.class_names = ['fire', 'smoke']  # ตรวจสอบชื่อคลาสให้ตรงกับโมเดล

    def detect(self, frame):
        results = self.model.predict(frame, verbose=False)
        return results[0].boxes

def process_videos(args):
    # ตรวจสอบ paths
    if not os.path.exists(args.model):
        print(f"ไม่พบไฟล์โมเดล: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"ไม่พบโฟลเดอร์วิดีโอ: {args.input}")
        return
    
    os.makedirs(args.output, exist_ok=True)
    
    # โหลดโมเดล
    print("กำลังโหลดโมเดล YOLOv12x...")
    detector = YOLOv12xDetector(args.model, args.conf)
    
    # ประมวลผลวิดีโอ
    video_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        input_path = os.path.join(args.input, video_file)
        output_path = os.path.join(args.output, f"detected_{video_file}")
        
        print(f"\nกำลังประมวลผล: {video_file}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ไม่สามารถเปิดวิดีโอ: {video_file}")
            continue
            
        # กำหนดคุณสมบัติวิดีโอ
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # ทำการตรวจจับ
            boxes = detector.detect(frame)
            
            # วาดผลลัพธ์
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                cls_id = int(box.cls.item())
                label = f"{detector.class_names[cls_id]} {conf:.2f}"
                color = (0, 0, 255) if detector.class_names[cls_id] == 'fire' else (0, 255, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            out.write(frame)
            
        cap.release()
        out.release()
        print(f"บันทึกผลลัพธ์ที่: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv12x Fire Detection')
    parser.add_argument('--model', required=True, help='พาธไปยังโมเดล best12.pt')
    parser.add_argument('--input', default='videos', help='โฟลเดอร์วิดีโออินพุต')
    parser.add_argument('--output', default='output', help='โฟลเดอร์ผลลัพธ์')
    parser.add_argument('--conf', type=float, default=0.5, help='ค่า Confidence threshold')
    
    args = parser.parse_args()
    
    process_videos(args)