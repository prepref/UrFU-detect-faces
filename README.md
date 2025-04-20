# UrFU-detect-faces
## Порядок запуска скриптов
1. extract_images.py
2. detect_faces.py
3. create_yolo_data.py
4. classifaication.py 

## Запуск fastapi
* В командной строке ввести uvicorn main:app --reload
* Перейти по ссылке http://127.0.0.1:8000/docs
* Указать в параметрах путь до UnRAR.exe
* Загрузить файл с изображениями .rar 

## Требования к .rar:
* Структура: Папка - Папки (каждая отдельный человек) - Изображения (.png, .jpg, .jpeg)