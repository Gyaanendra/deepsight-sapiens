import cv2
import os
import random
import json
import matplotlib.pyplot as plt

# --- CONFIG PATHS (Verify these match your local folders) ---
ORIG_IMG_DIR = r"E:\deepsight\train set-20260424T113859Z-3-001\train set\images"
ORIG_LBL_DIR = r"E:\deepsight\train set-20260424T113859Z-3-001\train set\labels"
AUG_BASE_DIR = "KnightSight_Local_Augmented" # The folder created by your previous script

def get_img_with_yolo_box(img_name):
    img_path = os.path.join(AUG_BASE_DIR, "images", img_name)
    txt_path = os.path.join(AUG_BASE_DIR, "labels", img_name.replace('.jpg', '.txt'))
    
    img = cv2.imread(img_path)
    if img is None: 
        return None
    
    H, W, _ = img.shape
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                # YOLO format: cls, xc, yc, w, h
                xc, yc, w, h = [float(x) for x in parts[1:]]
                # Convert YOLO normalized to Pixel for drawing
                x1 = int((xc - w/2) * W)
                y1 = int((yc - h/2) * H)
                x2 = int((xc + w/2) * W)
                y2 = int((yc + h/2) * H)
                # Green Box for Augmented
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    return img

def run_visual_audit():
    # 1. Pick 10 random base images from the original set
    all_originals = [f for f in os.listdir(ORIG_IMG_DIR) if f.endswith('.jpg')]
    sample_files = random.sample(all_originals, 10)
    
    # 2. Setup Plot (10 rows, 3 columns)
    fig, axes = plt.subplots(10, 3, figsize=(18, 50))
    
    for idx, f_name in enumerate(sample_files):
        base_name = os.path.splitext(f_name)[0]
        
        # --- COLUMN 1: ORIGINAL (RAW) ---
        orig_img = cv2.imread(os.path.join(ORIG_IMG_DIR, f_name))
        with open(os.path.join(ORIG_LBL_DIR, base_name + ".json"), 'r') as f:
            d = json.load(f)[0]
        # Blue Box for Original
        cv2.rectangle(orig_img, (int(d['x']), int(d['y'])), 
                      (int(d['x']+d['width']), int(d['y']+d['height'])), (255, 0, 0), 2)
        
        # --- COLUMN 2: AUG VARIATION 0 (Night/Low Light) ---
        v0_img = get_img_with_yolo_box(f"{base_name}_v0.jpg")
        
        # --- COLUMN 3: AUG VARIATION 1 (Extreme Glare/CLAHE) ---
        v1_img = get_img_with_yolo_box(f"{base_name}_v1.jpg")

        # Display Logic
        axes[idx, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f"Raw: {f_name}", fontsize=10)
        
        if v0_img is not None:
            axes[idx, 1].imshow(cv2.cvtColor(v0_img, cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title("V0: Night Simulation", fontsize=10)
        else:
            axes[idx, 1].text(0.5, 0.5, 'V0 Missing', ha='center')
        
        if v1_img is not None:
            axes[idx, 2].imshow(cv2.cvtColor(v1_img, cv2.COLOR_BGR2RGB))
            axes[idx, 2].set_title("V1: Glare/CLAHE", fontsize=10)
        else:
            axes[idx, 2].text(0.5, 0.5, 'V1 Missing', ha='center')
        
        for ax in axes[idx]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_visual_audit()