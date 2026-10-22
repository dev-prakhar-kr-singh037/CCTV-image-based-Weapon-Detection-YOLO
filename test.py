from ultralytics import YOLO

# 1. Load your trained model weights from Google Drive
model_path = '/content/drive/MyDrive/YOLOv8_Weapon_Detection/yolov8_run-3/weights/best.pt'  # Update subfolder path if stored in a specific folder inside MyDrive
model = YOLO(model_path)

# 2. Validate model on the test split using the yaml file
metrics = model.val(data='/content/data_colab.yaml', split='test')

# Print summary metrics
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50:    {metrics.box.map50}")
print(f"Precision: {metrics.box.p.mean()}")
print(f"Recall:    {metrics.box.r.mean()}")

'''# 3. Run prediction on the test images folder
results = model.predict(
    source="/content/dataset/dataset_combined/test/images",
    save=True,
    conf=0.25
)'''

#3. Run prediction on the test image folder and saving the output permanently
results = model.predict(
    source=os.path.join(dataset_root, 'test/images'),
    save=True,
    conf=0.25,
    project='/content/drive/MyDrive/YOLO_Outputs',  # Main Drive directory
    name='test_results'                             # Subfolder name
)
