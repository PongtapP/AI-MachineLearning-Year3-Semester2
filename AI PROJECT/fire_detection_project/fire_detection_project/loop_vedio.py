import cv2
import os

def play_video_loop(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ไม่สามารถเปิดวิดีโอ: {video_path}")
        return
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # กลับไปเฟรมแรก
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Detected Video", frame)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_folder = "output"  # โฟลเดอร์ที่เก็บวิดีโอที่ประมวลผลแล้ว
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    
    if not video_files:
        print("ไม่พบวิดีโอในโฟลเดอร์ output")
    else:
        for video in video_files:
            play_video_loop(os.path.join(video_folder, video))