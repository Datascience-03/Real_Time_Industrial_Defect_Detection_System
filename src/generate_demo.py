import os
import time
import cv2
from ultralytics import YOLO

def main():
    model_path = "runs/detect/train/weights/best.pt"
    video_path = "dataset/vedios/sample1.mp4"
    output_video_path = "outputs/demo_video.mp4"
    screenshot_dir = "outputs/sample_results"
    
    os.makedirs(screenshot_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
        
    print(f"Loading YOLO model: {model_path}...")
    model = YOLO(model_path)
    
    print(f"Opening video source: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Properties: Resolution={width}x{height}, FPS={fps_in:.2f}, Total Frames={total_frames}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_in, (width, height))
    
    frame_count = 0
    screenshot_count = 0
    max_screenshots = 3
    
    print("Processing video frames and running inference...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        t0 = time.time()
        
        # Run inference (using confidence threshold of 0.3 to catch defects)
        results = model(frame, conf=0.3, verbose=False)
        t1 = time.time()
        
        # Calculate inference-only FPS and drawing latency
        latency_ms = (t1 - t0) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        
        # Draw predictions on frame
        annotated_frame = frame.copy()
        has_detection = False
        
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                has_detection = True
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red box
                # Draw label background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 255), -1)
                # Draw text
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
        # Overlay FPS on the frame
        fps_text = f"Inference FPS: {fps:.1f} ({latency_ms:.1f} ms/frame)"
        cv2.putText(
            annotated_frame,
            fps_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0), # Green text
            2,
            cv2.LINE_AA
        )
        
        # Write the annotated frame to output video
        out.write(annotated_frame)
        
        # Save up to 3 screenshots when defects are detected (spread across the video)
        if has_detection and screenshot_count < max_screenshots:
            # We want to avoid capturing consecutive frames, so let's space them out
            # by at least 15 frames or check if the frame index matches a spread.
            # Let's save on frame detection if we haven't saved for a bit.
            screenshot_path = os.path.join(screenshot_dir, f"screenshot_{screenshot_count + 1}.jpg")
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Saved screenshot {screenshot_count + 1} at frame {frame_count} to {screenshot_path}")
            screenshot_count += 1
            
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")
            
    cap.release()
    out.release()
    
    print("\nProcessing complete!")
    print(f"Saved annotated demo video to '{output_video_path}'")
    print(f"Saved {screenshot_count} screenshots in '{screenshot_dir}'")

if __name__ == "__main__":
    main()
