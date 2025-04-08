import os

from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def create_dataset_files(dataset_path, train_ratio=0.8, random_state=42):
    classes = sorted(os.listdir(dataset_path))
    images = []
    labels = []

    for class_id, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            images.append(img_path)
            labels.append(class_id)

    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, train_size=train_ratio, random_state=random_state
    )

    with open('./images/train.txt', 'w') as f:
        for img_path, label in zip(train_images, train_labels):
            f.write(f"{img_path} {label}\n")

    with open('./images/val.txt', 'w') as f:
        for img_path, label in zip(val_images, val_labels):
            f.write(f"{img_path} {label}\n")

    return classes

def create_yaml_config(classes, yaml_path='./images/data.yaml'):
    """
    Создает YAML-конфиг для YOLO.
    
    Args:
        classes (list): Список имен классов.
        yaml_path (str): Куда сохранить конфиг.
    """
    config = {
        'train': 'train.txt',
        'val': 'val.txt',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(yaml_path, 'w') as f:
        f.write(f"train: {config['train']}\n")
        f.write(f"val: {config['val']}\n")
        f.write("names:\n")
        for i, name in enumerate(classes):
            f.write(f"  {i}: {name}\n")
    
    return None

create_yaml_config(['vadik', 'vasya'])

def train_yolo_classifier(yaml_path='./images/data.yaml', model_name='yolo11n-cls.pt',
                        epochs=50, imgsz=640, batch=16):
    """
    Обучает классификатор YOLOv8.
    
    Args:
        yaml_path (str): Путь к YAML-конфигу.
        model_name (str): Модель для обучения.
        epochs (int): Количество эпох.
        imgsz (int): Размер изображения.
        batch (int): Размер батча.
    """
    model = YOLO(model_name)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        augment=True  # Аугментация данных
    )
    return results