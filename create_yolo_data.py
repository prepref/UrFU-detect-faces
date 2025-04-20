import os
import shutil
from sklearn.model_selection import train_test_split


def prepare_yolo_dataset(input_dir, output_dir, train_ratio=0.8):
    """Подготавливает датасет для YOLO"""
    # Создаем директории
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Получаем классы
    classes = [d for d in os.listdir(input_dir) 
              if os.path.isdir(os.path.join(input_dir, d))]
    
    if not classes:
        raise ValueError("No classes found in input directory")

    # Собираем и разделяем изображения
    samples = []
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        for img in os.listdir(cls_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(cls_dir, img), cls))

    if not samples:
        raise ValueError("No images found for training")

    train, val = train_test_split(samples, train_size=train_ratio)

    # Копируем файлы
    for split, target_dir in [(train, train_dir), (val, val_dir)]:
        for img_path, cls in split:
            cls_dir = os.path.join(target_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy(img_path, cls_dir)

    # Создаем конфиг
    yaml_content = f"""train: {train_dir}
val: {val_dir}
nc: {len(classes)}
names: {classes}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)