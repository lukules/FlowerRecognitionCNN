import os
import hashlib

def calculate_hash(file_path):
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(4096):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def remove_duplicates(folder_path):
    hash_dict = {}
    duplicates_removed = 0

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_hash = calculate_hash(file_path)

            if file_hash in hash_dict:
                print(f'Removing duplicate: {filename}')
                os.remove(file_path)
                duplicates_removed += 1
            else:
                hash_dict[file_hash] = file_path

    print(f'Duplicate removal completed. Removed {duplicates_removed} duplicates.')

main_folder_path = r'C:\Users\kules\Desktop\SzUM'

subfolders = ['tulip', 'daisy', 'dandelion', 'sunflower', 'rose']

for subfolder in subfolders:
    subfolder_path = os.path.join(main_folder_path, subfolder)
    print(f'Removing duplicates in folder: {subfolder}')
    remove_duplicates(subfolder_path)
