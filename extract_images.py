import rarfile
import os


def extract_images(rar_path: str, dist_path: str, rarfile_tool_path: str):
    c = 0
    rarfile.UNRAR_TOOL = rarfile_tool_path
    archive_name = os.path.splitext(os.path.basename(rar_path))[0]
    target_folder = os.path.join(dist_path, archive_name)
    os.makedirs(target_folder, exist_ok=True)

    with rarfile.RarFile(rar_path) as rf:
        for file in rf.namelist():
            if file.endswith("/"):
                continue
            
            if file.lower().endswith((".jpg", ".jpeg", ".png",)):

                ext = os.path.splitext(file)[-1].lower()

                new_name = f"image_{c}{ext}"
                c+=1
                
                output_path = os.path.join(dist_path, archive_name, new_name)
                
                with rf.open(file) as src, open(output_path, 'wb') as dst:
                        dst.write(src.read())

    print("Архив успешно распакован!")

extract_images("vasya.rar", "./images/source-images", "C:\\Different\\WinRAR\\UnRAR.exe")