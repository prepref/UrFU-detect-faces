import os
import shutil
from ultralytics import YOLO

def train_yolo_classifier(datasets_dir, runs_dir, model_path, output_path, **train_kwargs):
    """Обучает классификатор YOLO"""
    
    # Создаем необходимые директории
    os.makedirs(runs_dir, exist_ok=True)
    
    # Загружаем модель
    model = YOLO('yolov8n-cls.pt')
    
    # Обучаем с явным указанием всех путей
    results = model.train(
        data=datasets_dir,
        project=runs_dir,
        **train_kwargs
    )

    shutil.copy2(model_path, output_path)

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Trained model not found at {output_path}")