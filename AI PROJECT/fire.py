import cv2
import numpy as np
import time
from ultralytics import YOLO
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fire and Smoke Detection using YOLOv8")
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video filename')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load the YOLOv8 model
    print(f"Loading model {args.model}...")
    model = YOLO(args.model)

    # Open the webcam
    print(f"Opening webcam at index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {args.camera}")
        return

    # Optimize for Raspberry Pi - set lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Get video properties for saving
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # Set fixed FPS for output video (Raspberry Pi might not achieve higher FPS)

    # Initialize video writer if save option is enabled
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to {args.output}")

    # Define class names and colors (update these according to your model)
    class_names = {
        0: 'fire',
        1: 'smoke'
    }
    colors = {
        0: (0, 0, 255),   # Red for fire
        1: (128, 128, 128)  # Gray for smoke
    }

    # Variables for FPS calculation
    frame_count = 0
    fps_avg = 0
    fps_start_time = time.time()

    # Process frames
    print("Starting realtime detection. Press 'q' to quit.")
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame from webcam")
            break

        # Start time for individual frame processing
        start_time = time.time()

        # Perform detection
        results = model(frame, conf=args.conf, verbose=False)[0]

        # Process results
        detections = results.boxes.data.cpu().numpy()

        # Check if fire or smoke detected
        fire_detected = False
        smoke_detected = False

        # Draw bounding boxes and labels
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)

            # Get class name and color
            class_name = class_names.get(class_id, f"Class {class_id}")
            color = colors.get(class_id, (0, 255, 0))  # Default to green

            # Set detection flags
            if class_id == 0:  # fire
                fire_detected = True
            elif class_id == 1:  # smoke
                smoke_detected = True

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add alert text if fire or smoke detected
        if fire_detected:
            cv2.putText(frame, "FIRE DETECTED!", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if smoke_detected:
            cv2.putText(frame, "SMOKE DETECTED!", (10, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)

        # Calculate and display FPS
        frame_count += 1
        if frame_count >= 10:  # Update FPS every 10 frames
            fps_end_time = time.time()
            fps_avg = frame_count / (fps_end_time - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0

        fps_text = f"FPS: {fps_avg:.1f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display processing time for this frame
        proc_time = (time.time() - start_time) * 1000  # convert to ms
        proc_text = f"Processing time: {proc_time:.1f} ms"
        cv2.putText(frame, proc_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the result
        cv2.imshow("Fire and Smoke Detection (Realtime)", frame)

        # Write frame to output video if save option is enabled
        if args.save and writer:
            writer.write(frame)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add small delay to reduce CPU usage on Raspberry Pi
        time.sleep(0.01)

    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Detection stopped")

if __name__ == "__main__":
    main()