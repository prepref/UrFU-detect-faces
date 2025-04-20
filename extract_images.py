import rarfile
import os
from typing import List
import itertools

def extract(rar_path: str, dist_path: str, rarfile_tool_path: str) -> List[str]:
    """Извлекает изображения из RAR-архива"""
    rarfile.UNRAR_TOOL = rarfile_tool_path
    
    with rarfile.RarFile(rar_path) as rf:
        files = [f for f in rf.namelist() 
                if not f.endswith("/") and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if not files:
            return []
        
        archive_name = os.path.splitext(os.path.basename(rar_path))[0]
        counter = itertools.count(1)
        
        for file in files:
            parts = [p for p in file.split("/") if p]
            folder = parts[-2] if len(parts) > 1 else archive_name
            
            target_folder = os.path.join(dist_path, folder)
            os.makedirs(target_folder, exist_ok=True)
            
            ext = os.path.splitext(file)[-1].lower()
            new_name = f"image_{next(counter)}{ext}"
            output_path = os.path.join(target_folder, new_name)
            
            with rf.open(file) as src, open(output_path, 'wb') as dst:
                dst.write(src.read())