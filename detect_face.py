from mtcnn import MTCNN
import matplotlib.image as mpimg
import os

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
                    cropped = img[y:y+h, x:x+w]
                    mpimg.imsave(os.path.join(dst_dir, file_name), cropped)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")