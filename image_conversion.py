from PIL import Image
import os

def convert_to_tiff(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Skip if it's already a TIFF file
        if filename.lower().endswith((".tiff", ".tif")):
            print(f"ℹ️ Skipped (already TIFF): {filename}")
            continue

        try:
            with Image.open(file_path) as img:
                # Ensure RGB mode for consistency
                img = img.convert("RGB")
                
                # New TIFF file path
                tiff_path = os.path.splitext(file_path)[0] + ".tiff"
                
                # Save as TIFF
                img.save(tiff_path, "TIFF")
                os.remove(file_path)  # Optional: remove original file
                print(f"✅ Converted: {filename} → {os.path.basename(tiff_path)}")
        
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

# 📂 Replace with your folder path
convert_to_tiff(r"D:\Masrafe\Masrafe extra\reasearch\mogonet\figure")
