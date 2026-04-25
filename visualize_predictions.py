import json
import cv2
import os
from tqdm import tqdm

PREDICTIONS_FILE = "Sapines_II/predictions.json"
IMG_DIR = "test_img/test/images"
OUTPUT_DIR = "visualized_predictions"

def main():
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Error: {PREDICTIONS_FILE} not found. Run create_submission.py first.")
        return
        
    if not os.path.exists(IMG_DIR):
        print(f"Error: {IMG_DIR} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(PREDICTIONS_FILE, "r") as f:
        predictions = json.load(f)
        
    print(f"Visualizing {len(predictions)} images...")
    
    for img_name, data in tqdm(predictions.items(), desc="Drawing boxes"):
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        boxes = data.get("plate_bbox", [])
        
        # Draw each box
        for box in boxes:
            # Check if it's the [0,0,0,0] fallback which means no detection
            if box == [0, 0, 0, 0]:
                cv2.putText(img, "NO DETECTION", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                continue
                
            x1, y1, x2, y2 = box
            
            # Draw a thick green bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Add a label
            cv2.putText(img, "Plate", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
        out_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(out_path, img)
        
    print(f"\nDone! Check the '{OUTPUT_DIR}' folder to see all the images with their bounding boxes drawn.")

if __name__ == "__main__":
    main()
