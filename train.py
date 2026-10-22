from ultralytics import YOLO
import torch

# Load pre-trained YOLOv8 model (yolov8m.pt or yolov8m.pt)
model = YOLO("yolov8m.pt")

# Check for CUDA availability
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Training will proceed on CPU. Please ensure your Colab runtime type is set to GPU (e.g., T4 GPU).")

# Start training
results = model.train(
    data="/content/data_colab.yaml",
    epochs=50,             # Adjust epochs based on dataset size
    imgsz=640,             # Standard image resolution (640-balanced)(1280-more accurate but slower)
    batch=16,              # Standard batch size for T4 GPU
    device=0,              # Use GPU (device=0) if available
    project="/content/drive/MyDrive/YOLOv8_Weapon_Detection", # Save outputs directly to Drive
    name="yolov8_run"
)