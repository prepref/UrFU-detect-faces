from mtcnn import MTCNN
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def detect_images(source_path: str, dist_path: str):
    general_dir = source_path
    new_general_dir = dist_path
    detector = MTCNN()
    for dir_name in os.listdir(general_dir):
        origin_dir = general_dir + "/" + dir_name
        new_dir = new_general_dir + "/" + dir_name
        os.makedirs(new_dir, exist_ok=True)
        for file_name in os.listdir(origin_dir):
            origin_path = origin_dir+'/'+file_name
            origin_img = mpimg.imread(origin_path)
            result = detector.detect_faces(origin_img)
            if result:
                (x, y, width, height) = result[0]['box']
                cropped_img = origin_img[y:y+height, x:x+width]
                new_path = new_dir+'/'+file_name
                mpimg.imsave(new_path, cropped_img)
                print(f"Обработка фото завершена {origin_path}")
            else:
                print(f'Лицо не обнаружено на изображении {origin_path}')

detect_images("./images/source-images", "./images/crop-images")