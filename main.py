from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from extract_images import extract
from detect_face import detect_images
from create_yolo_data import prepare_yolo_dataset
from classification import train_yolo_classifier
import rarfile

app = FastAPI()

@app.post("/process-images/")
async def api_process_images(
    rar_file: UploadFile = File(...),
    unrar_tool_path: str = "C:\\Program Files\\WinRAR\\UnRAR.exe",
    epochs: int = 10,
    imgsz: int = 224,
    batch: int = 8
):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Извлечение изображений
            source_dir = os.path.join(temp_dir, "images", "source-images")
            crop_dir = os.path.join(temp_dir, "images", "crop-images")
            os.makedirs(source_dir, exist_ok=True)
            os.makedirs(crop_dir, exist_ok=True)
            
            rar_path = os.path.join(temp_dir, rar_file.filename)
            with open(rar_path, "wb") as f:
                f.write(await rar_file.read())

            extract(
                rar_path=rar_path,
                dist_path=source_dir,
                rarfile_tool_path=unrar_tool_path
            )
            
            extracted_files = []
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        extracted_files.append(os.path.join(root, file))

            if not extracted_files:
                raise HTTPException(status_code=400, detail="No images found in the archive")

            # 2. Обнаружение лиц
            detect_images(source_dir, crop_dir)
            processed_files = []
            for root, _, files in os.walk(crop_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        processed_files.append(os.path.join(root, file))

            if not processed_files:
                raise HTTPException(status_code=400, detail="No faces detected in images")
            
            # 3. Подготовка YOLO датасета
            output_dir = os.path.join(temp_dir, "datasets")
            prepare_yolo_dataset(crop_dir, output_dir)
            yolo_files = []
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        yolo_files.append(os.path.join(root, file))
            
            if not yolo_files:
                raise HTTPException(status_code=400, detail="Failed to create YOLO dataset")

            # 4. Обучение модели
            runs_dir = os.path.join(temp_dir, "runs")
            model_path = os.path.join(runs_dir, "train", "weights", "best.pt")
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
            train_yolo_classifier(
                datasets_dir=output_dir,
                runs_dir=runs_dir,
                model_path=model_path,
                output_path=output_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                augment=False
            )

            # Возвращаем модель для скачивания
            return JSONResponse({
                "status": "success",
                "path_to_model": output_path
            })

    except rarfile.BadRarFile:
        raise HTTPException(status_code=400, detail="Invalid RAR archive format")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))