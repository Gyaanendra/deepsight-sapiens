from ultralytics import YOLO

# 1. Load your best trained model
model_path = r"E:\deepsight\runs\detect\KnightSight_Local\RTX4060_Uniform_v1\weights\best.pt"
model = YOLO(model_path)

# 2. Export to ONNX format
# ONNX is highly optimized for CPU inference (Colab / Edge devices)
print("Exporting model to ONNX...")
model.export(format="onnx", imgsz=480, simplify=True)

print("Export Complete! Check the weights folder for 'best.onnx'")
