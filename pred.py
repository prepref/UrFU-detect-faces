from ultralytics import YOLO
from mtcnn import MTCNN
import matplotlib.image as mpimg
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def detect_images(source_path, dist_path):
    """Обнаруживает лица и сохраняет обрезанные изображения"""
    detector = MTCNN()
    
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    os.makedirs(dist_path, exist_ok=True)
    
    for dir_name in os.listdir(source_path):
        src_dir = os.path.join(source_path, dir_name)
        dst_dir = os.path.join(dist_path, dir_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        for file_name in os.listdir(src_dir):
            if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(src_dir, file_name)
            try:
                img = mpimg.imread(img_path)
                faces = detector.detect_faces(img)
                
                if faces:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    cropped = img[y-3:y+h+3, x-3:x+w+3]
                    mpimg.imsave(os.path.join(dst_dir, file_name), cropped)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")


def classify(image_path, model):
    """Предсказание класса для одного изображения"""
    results = model(image_path, imgsz=224)
    probs = results[0].probs
    top1 = probs.top1
    return top1

def evaluate_folder(folder_path, model, true_labels):
    """Оценка всех изображений в папке"""
    pred_labels = []
    
    for dir_name in os.listdir(folder_path):
        dir_name = os.path.join(folder_path, dir_name)
        for filename in os.listdir(dir_name):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dir_name, filename)
                pred = classify(img_path, model)
                pred_labels.append(pred)
        
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    print(true_labels)
    print(pred_labels)
    
    return accuracy, f1

def main():
    model = YOLO("best1.pt")
    
    test_folder = "./test/"
    detect_images("./test/", "./res-test/")

    results_folder = "./res-test/"
    
    true_labels = []
    class_names = os.listdir(test_folder)
    class_map = [2,3,4]
    class_to_idx = {name: class_map[i] for i, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_folder = os.path.join(results_folder, class_name)
        if os.path.isdir(class_folder):
            num_images = len([f for f in os.listdir(class_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            true_labels.extend([class_to_idx[class_name]] * num_images)
    
    accuracy, f1 = evaluate_folder(results_folder, model, true_labels)
    
    print(f"\nРезультаты оценки:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return None

if __name__ == "__main__":
    main()