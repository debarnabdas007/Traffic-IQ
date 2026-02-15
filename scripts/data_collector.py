import cv2
import csv
import os
import sys

# --- CONFIGURATION ---
# CHANGE THIS NAME to switch between your different video files
VIDEO_FILENAME = 'video6.mp4' 

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project Root
VIDEO_PATH = os.path.join(BASE_DIR, 'data', 'raw', VIDEO_FILENAME)
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'vehicle_data.csv')

# Tunable Parameters
MIN_AREA = 250      # Increased slightly to ignore small noise
SKIP_FRAMES = 2      # Process 1 out of every 3 frames
JUMP_MINUTES = 2     # Jump 2 mins at a time (better for shorter clips)

def init_csv():
    # Ensure directory exists
    if not os.path.exists(os.path.dirname(DATASET_PATH)):
        os.makedirs(os.path.dirname(DATASET_PATH))
    
    # Create file if missing (Append mode is safe)
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Area', 'Aspect_Ratio', 'Label'])
            print(f"Created new dataset: {DATASET_PATH}")
    else:
        print(f"Found existing dataset: {DATASET_PATH} (Appending new data)")

def collect_data():
    print(f"DEBUG: Loading video from: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print("❌ ERROR: Video file not found!")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ ERROR: Could not open video source.")
        sys.exit(1)

    # Get Video Properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize MOG2
    object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    print(f"--- Loaded: {VIDEO_FILENAME} ({total_frames/fps/60:.1f} mins) ---")
    print("KEYS: [c]=Car, [t]=Truck, [b]=Bike")
    print("NAV:  [j]=Jump Time, [space]=Pause/Resume")
    print("OPS:  [s]=Skip, [q]=Quit")

    stabilize_counter = 0 

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Fast scanning
        if stabilize_counter == 0 and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % SKIP_FRAMES != 0:
            continue

        # 1. Resize (Standardize input)
        frame = cv2.resize(frame, (1020, 600)) 
        
        # 2. Apply MOG2 (Get the mask)
        mask = object_detector.apply(frame)
        
        # 3. THE GLUE LOGIC (Morphological Operations)
        # Step A: Remove Shadows (Gray pixels)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        # Step B: Define the "Brush" (Kernel) - 5x5 pixel square
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) 

        # Step C: Close holes (Fill gaps inside objects like windshields)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Step D: Dilation (Make objects fatter to merge close parts)
        mask = cv2.dilate(mask, kernel, iterations=2) 
        # ---------------------------------------------------------

        # Stabilization Logic (Run MOG2 blindly after a jump)
        if stabilize_counter > 0:
            stabilize_counter -= 1
            cv2.putText(frame, "STABILIZING BACKGROUND...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Labeling Station", frame)
            cv2.waitKey(1)
            continue

        # 4. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        valid_cnt = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                valid_cnt = cnt
                break # Pause on the first valid object

        cv2.imshow("Labeling Station", frame)

        # Wait ONLY if we found a valid object
        key = cv2.waitKey(0 if valid_cnt is not None else 1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('j'): # TIME JUMP
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            jump_frames = JUMP_MINUTES * 60 * fps
            new_pos = min(current_frame + jump_frames, total_frames - 100)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            # Reset MOG2 for new lighting
            object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
            stabilize_counter = 40 # Re-learn for 40 frames
            print(f"Jumped to {int(new_pos)}. Stabilizing...")
            continue

        label = None
        if valid_cnt is not None:
            area = cv2.contourArea(valid_cnt)
            x, y, w, h = cv2.boundingRect(valid_cnt)
            aspect_ratio = float(w) / h
            
            if key == ord('c'): label = 0
            elif key == ord('t'): label = 1
            elif key == ord('b'): label = 2
            
            if label is not None:
                print(f"Saved Class {label} (Area: {area:.0f}, AR: {aspect_ratio:.2f})")
                with open(DATASET_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([area, aspect_ratio, label])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_csv()
    collect_data()