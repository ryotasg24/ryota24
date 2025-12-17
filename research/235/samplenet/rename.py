import os

# クラスの対応表
class_mapping = {
    'class_0': 'airplane', 'class_1': 'bathtub', 'class_2': 'bed', 'class_3': 'bench',
    'class_4': 'bookshelf', 'class_5': 'bottle', 'class_6': 'bowl', 'class_7': 'car',
    'class_8': 'chair', 'class_9': 'cone', 'class_10': 'cup', 'class_11': 'curtain',
    'class_12': 'desk', 'class_13': 'door', 'class_14': 'dresser', 'class_15': 'flower_pot',
    'class_16': 'glass_box', 'class_17': 'guitar', 'class_18': 'keyboard', 'class_19': 'lamp',
    'class_20': 'laptop', 'class_21': 'mantel', 'class_22': 'monitor', 'class_23': 'night_stand',
    'class_24': 'person', 'class_25': 'piano', 'class_26': 'plant', 'class_27': 'radio',
    'class_28': 'range_hood', 'class_29': 'sink', 'class_30': 'sofa', 'class_31': 'stairs',
    'class_32': 'stool', 'class_33': 'table', 'class_34': 'tent', 'class_35': 'toilet',
    'class_36': 'tv_stand', 'class_37': 'vase', 'class_38': 'wardrobe', 'class_39': 'xbox'
}

def rename_directories_and_files(base_dir):
    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)

        if os.path.isdir(subdir_path) and subdir_name.startswith('class_'):
            # Rename subdirectories
            new_subdir_name = class_mapping.get(subdir_name, subdir_name)
            new_subdir_path = os.path.join(base_dir, new_subdir_name)
            if subdir_path != new_subdir_path:
                os.rename(subdir_path, new_subdir_path)

            # Rename files within the subdirectory
            for filename in os.listdir(new_subdir_path):
                file_path = os.path.join(new_subdir_path, filename)
                new_filename = f"{new_subdir_name}_{filename.split('_')[-1]}"
                new_file_path = os.path.join(new_subdir_path, new_filename)
                if file_path != new_file_path:
                    os.rename(file_path, new_file_path)

            # Sort files within the subdirectory
            txt_files = [f for f in os.listdir(new_subdir_path) if f.endswith('.txt')]
            txt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

            for i, txt_file in enumerate(txt_files, start=1):
                new_txt_file = f"{new_subdir_name}_{i}.txt"
                os.rename(os.path.join(new_subdir_path, txt_file), os.path.join(new_subdir_path, new_txt_file))

if __name__ == "__main__":
    base_directory = "loss10_SampleNet50/eval/class_data"  
    rename_directories_and_files(base_directory)

