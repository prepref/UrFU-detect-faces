import os
import shutil

from sklearn.model_selection import train_test_split
from ultralytics import YOLO, settings

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

settings.update({
    'datasets_dir': os.path.join(PROJECT_ROOT, 'datasets'),  # Все датасеты будут здесь
    'weights_dir': os.path.join(PROJECT_ROOT, 'weights'),    # Для сохранения моделей
    'runs_dir': os.path.join(PROJECT_ROOT, 'runs')          # Для результатов тренировок
})

def prepare_yolo_dataset(input_dir, output_dir='datasets', train_ratio=0.8, random_state=42):
    """
    Подготавливает датасет для YOLO с заданной структурой
    
    Args:
        input_dir (str): Путь к папке с классами (каждая подпапка - отдельный класс)
        output_dir (str): Основная директория для выходных данных (по умолчанию 'dataset')
        train_ratio (float): Доля данных для обучения (по умолчанию 0.8)
        random_state (int): Seed для воспроизводимости разбиения
    """
    # Создаем структуру директорий
    train_img_dir = os.path.join(output_dir, 'train')
    val_img_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    # Получаем список классов (по именам подпапок)
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    class_to_id = {cls: idx for idx, cls in enumerate(classes)}
    
    # Собираем все изображения с путями и метками
    samples = []
    for class_name in classes:
        class_dir = os.path.join(input_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, class_name))

    # Разделяем на train/val
    train_samples, val_samples = train_test_split(
        samples, train_size=train_ratio, random_state=random_state
    )

    # Функция для обработки split'а
    def process_split(samples, split_dir):
        # Создаем подпапки для каждого класса
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
        
        # Копируем изображения в соответствующие подпапки
        for img_path, class_name in samples:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(split_dir, class_name, img_name)
            shutil.copy(img_path, dst_path)

    # Обрабатываем train и val
    process_split(train_samples, train_img_dir)
    process_split(val_samples, val_img_dir)

    # Создаем data.yaml
    yaml_content = f"""train: train
val: val
nc: {len(classes)}
names: {classes}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    print(f"Датасет подготовлен. Классы: {classes}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")

# prepare_yolo_dataset(input_dir="./images/crop-images/")

def train_yolo_classifier(yaml_path='data.yaml', model_name='yolo11n-cls.pt',
                        epochs=10, imgsz=225, batch=8):
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
        data="datasets",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        augment=False  # Аугментация данных
    )
    return results

train_yolo_classifier()
