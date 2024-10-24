import os
from PIL import Image


main_folder = r'C:\Users\kules\Desktop\SzUM'

target_size = (128, 128)

skipped_files_count = 0

for folder_name in ['sunflower', 'tulip', 'daisy', 'dandelion', 'rose']:
    input_folder = os.path.join(main_folder, folder_name)
    output_folder = os.path.join(main_folder, f'{folder_name}_128')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)

            if img.mode != 'RGB':
                skipped_files_count += 1
                print(f'Skipped {filename} from {folder_name}: Not in RGB mode.')
                continue

            img_resized = img.resize(target_size)

            output_path = os.path.join(output_folder, filename)

            img_resized.save(output_path)

print('Processing completed.')
print(f'Total skipped files: {skipped_files_count}')
