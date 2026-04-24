import os
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import easyocr
except ImportError as exc:
    raise ImportError(
        "easyocr is required for OCR. Install it with `pip install easyocr`"
    ) from exc

MODEL_PATH = r"E:\deepsight\runs\detect\KnightSight_Local\RTX4060_Uniform_v1\weights\best.pt"
IMAGE_PATH = r"E:\deepsight\KnightSight_Uniform_Dataset\images\0a70a4b1-65c0-49d4-a09a-ec62b9fe20ca_v1.jpg"
CONFIDENCE = 0.25
IMG_SIZE = 640
CLASS_NAMES = {0: "license_plate"}


def annotate_image(image, boxes, classes, scores, ocr_texts, class_names):
    annotated = image.copy()
    for (x1, y1, x2, y2), cls, score, ocr in zip(boxes, classes, scores, ocr_texts):
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cls, 'plate')} {score:.2f}"
        if ocr:
            label += f" | {ocr}"

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_w, text_h = text_size
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 12, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def ocr_from_box(image, box, reader):
    x1, y1, x2, y2 = box
    region = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
    if region.size == 0:
        return ""

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )

    text_results = reader.readtext(thresh, detail=0, paragraph=False)
    return " ".join([t.strip() for t in text_results if t.strip()])


def run_inference(image_path=IMAGE_PATH, model_path=MODEL_PATH, conf=CONFIDENCE, imgsz=IMG_SIZE, device=0):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    model = YOLO(model_path)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")

    results = model.predict(image_path, conf=conf, imgsz=imgsz, device=device)
    result = results[0]
    boxes = result.boxes

    if len(boxes) == 0:
        print("No detections found.")
        return

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    scores = boxes.conf.cpu().numpy().astype(float)
    classes = boxes.cls.cpu().numpy().astype(int)
    ocr_texts = [ocr_from_box(image, box, reader) for box in xyxy]

    print("Detected plates:")
    for idx, (cls, score, ocr_text) in enumerate(zip(classes, scores, ocr_texts), start=1):
        print(f" {idx}. {CLASS_NAMES.get(cls, str(cls))} | conf={score:.2f} | OCR='{ocr_text}'")

    annotated = annotate_image(image, xyxy, classes, scores, ocr_texts, CLASS_NAMES)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("YOLO detection + OCR")
    plt.show()

    for idx, box in enumerate(xyxy, start=1):
        x1, y1, x2, y2 = box
        crop = image[max(0, y1):min(image.shape[0], y2), max(0, x1):min(image.shape[1], x2)]
        if crop.size == 0:
            continue
        plt.figure(figsize=(6, 3))
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Crop {idx}: OCR = {ocr_texts[idx-1]}")
        plt.show()


if __name__ == "__main__":
    run_inference()
