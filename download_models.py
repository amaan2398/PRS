# 🚀 Deployed App: https://quiet-cove-96035-14ffb6e5a72f.herokuapp.com/
# 📂 GitHub Repo: https://github.com/amaan2398/PRS
import gdown
import os
import shutil
import sys

def download_models():
    # Folder link provided by user
    url = 'https://drive.google.com/drive/folders/1OwcKTEwWPvNT4ADXbgFXR8ZoA26lhKmG?usp=sharing'
    
    # We want to place files in 'models/' directory
    # We'll download to a temp directory first to inspect structure
    temp_dir = 'temp_models_download'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print("Downloading models from Google Drive...")
    try:
        files = gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"Error downloading folder: {e}")
        sys.exit(1)
    
    if not files:
        print("No files downloaded. Please check the Google Drive link.")
        # We don't exit here because maybe gdown returned empty list but files are there? 
        # Unlikely.
    
    print(f"Downloaded {len(files) if files else 0} files.")
    
    # Inspect temp_dir
    # gdown usually creates the folder name inside temp_dir
    # e.g. temp_dir/MyFolder/...
    # or sometimes directly in temp_dir if it can't determine name?
    # Let's check subdirectories
    items = os.listdir(temp_dir)
    source_dir = temp_dir
    
    # If there is exactly one directory and it looks like the drive folder
    if len(items) == 1 and os.path.isdir(os.path.join(temp_dir, items[0])):
        source_dir = os.path.join(temp_dir, items[0])
        print(f"Found inner folder: {items[0]}")
    
    target_base = 'models'
    if not os.path.exists(target_base):
        os.makedirs(target_base)
        
    print(f"Moving files to {target_base}...")
    
    # Move contents
    for item in os.listdir(source_dir):
        s = os.path.join(source_dir, item)
        d = os.path.join(target_base, item)
        
        if os.path.isdir(s):
            # If it's a directory (e.g. recommendation), merge it
            if os.path.exists(d):
                # If target exists, we merge/overwrite files inside
                for subitem in os.listdir(s):
                    sub_s = os.path.join(s, subitem)
                    sub_d = os.path.join(d, subitem)
                    if os.path.exists(sub_d):
                        os.remove(sub_d)
                    shutil.move(sub_s, sub_d)
                shutil.rmtree(s) # Remove source dir after moving contents
            else:
                shutil.move(s, d)
        else:
            # File
            if os.path.exists(d):
                os.remove(d)
            shutil.move(s, d)
            
        print(f"Synced {item}")

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print("Model download and setup complete.")

if __name__ == "__main__":
    download_models()
