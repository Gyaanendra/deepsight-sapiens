import albumentations as A
import cv2
import json
import os
from tqdm import tqdm
from multiprocessing import Pool

# --- CONFIG ---
BASE_DIR = r"E:\deepsight\train set-20260424T113859Z-3-001\train set"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
OUT_DIR = "KnightSight_Uniform_Dataset"

IMG_OUT = os.path.join(OUT_DIR, "images")
LBL_OUT = os.path.join(OUT_DIR, "labels")
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# TARGET DIMENSIONS
TARGET_W, TARGET_H = 472, 303

# AUGMENTATION PIPELINES
# Low Light: Darkened with "Micro-Grain"
night_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.3), contrast_limit=0.1, p=1.0),
    # var_limit (2, 8) is barely visible but mathematically presence
    # p=0.4 means not every night image is grainy
    A.GaussNoise(var_limit=(2, 8), p=0.4), 
    A.Blur(blur_limit=3, p=0.3) # Adding a tiny blur to simulate low-light focus
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

glare_transform = A.Compose([
    A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(0.1, 0.15), contrast_limit=0.2, p=1.0),
    A.Sharpen(alpha=(0.1, 0.3), p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

def process_single_image(f_name):
    try:
        img_path = os.path.join(IMG_DIR, f_name)
        img = cv2.imread(img_path)
        if img is None: return
        
        # --- UNIFORMITY CHECK ---
        h, w, _ = img.shape
        if w != TARGET_W or h != TARGET_H:
            return # Skip non-conforming images
        
        lbl_path = os.path.join(LBL_DIR, f_name.replace('.jpg', '.json'))
        if not os.path.exists(lbl_path): return
        
        with open(lbl_path, 'r') as f:
            d = json.load(f)[0]
        
        bboxes = [[d['x'], d['y'], d['width'], d['height']]]
        base_name = os.path.splitext(f_name)[0]

        # --- V0: ORIGINAL (Formatted for YOLO) ---
        xc, yc = (d['x'] + d['width']/2)/TARGET_W, (d['y'] + d['height']/2)/TARGET_H
        nw, nh = d['width']/TARGET_W, d['height']/TARGET_H
        cv2.imwrite(os.path.join(IMG_OUT, f"{base_name}_v0.jpg"), img)
        with open(os.path.join(LBL_OUT, f"{base_name}_v0.txt"), 'w') as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

        # --- V1: LOW LIGHT ---
        t1 = night_transform(image=img, bboxes=bboxes, class_labels=[0])
        if t1['bboxes']:
            bx, by, bw, bh = t1['bboxes'][0]
            n_xc, n_yc = (bx + bw/2)/TARGET_W, (by + bh/2)/TARGET_H
            n_w, n_h = bw/TARGET_W, bh/TARGET_H
            cv2.imwrite(os.path.join(IMG_OUT, f"{base_name}_v1.jpg"), t1['image'])
            with open(os.path.join(LBL_OUT, f"{base_name}_v1.txt"), 'w') as f:
                f.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}\n")

        # --- V2: EXTREME GLARE ---
        t2 = glare_transform(image=img, bboxes=bboxes, class_labels=[0])
        if t2['bboxes']:
            bx, by, bw, bh = t2['bboxes'][0]
            n_xc, n_yc = (bx + bw/2)/TARGET_W, (by + bh/2)/TARGET_H
            n_w, n_h = bw/TARGET_W, bh/TARGET_H
            cv2.imwrite(os.path.join(IMG_OUT, f"{base_name}_v2.jpg"), t2['image'])
            with open(os.path.join(LBL_OUT, f"{base_name}_v2.txt"), 'w') as f:
                f.write(f"0 {n_xc:.6f} {n_yc:.6f} {n_w:.6f} {n_h:.6f}\n")

    except Exception as e:
        return f"Error {f_name}: {e}"

if __name__ == '__main__':
    files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    print(f"🧐 Filtering and Processing {len(files)} potential images...")
    with Pool(8) as p:
        list(tqdm(p.imap_unordered(process_single_image, files), total=len(files)))
    
    # Final count check
    final_count = len(os.listdir(IMG_OUT))
    print(f"✅ Success! Generated {final_count} total images ({final_count//3} unique scenes).")