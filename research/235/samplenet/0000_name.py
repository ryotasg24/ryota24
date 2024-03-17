import os

def rename_files(root_directory):
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".txt"):
                parts = file.split('_')
                if len(parts) >= 3:
                    class_name, file_number = '_'.join(parts[:-1]), int(parts[-1].split('.')[0])
                else:
                    class_name, file_number = parts[0], int(parts[1].split('.')[0])

                new_file_number = str(file_number).zfill(4)
                new_filename = f"{class_name}_{new_file_number}.txt"

                current_path = os.path.join(subdir, file)
                new_path = os.path.join(subdir, new_filename)

                os.rename(current_path, new_path)
                print(f"Renamed: {file} to {new_filename}")


root_directory = '/pointnet2/samplenet/classification/log/loss10_SampleNet50/eval/class_data'


rename_files(root_directory)

