import os
import glob

def save_below_n_files(directory, threshold):
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    below_threshold_files = []

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory, subdirectory)

        txt_files = glob.glob(os.path.join(subdirectory_path, '*.txt'))

        if not txt_files:
            print(f"No txt files found in {subdirectory_path}")
            continue

        for txt_file in txt_files:
            with open(txt_file, 'r') as file:
                lines = sum(1 for line in file)

                if lines <= threshold:
                    below_threshold_files.append(txt_file)

    if below_threshold_files:
        print(f"Files with {threshold} or fewer lines:")
        for file in below_threshold_files:
            print(file)
        output_path = os.path.join(directory, f'below_{threshold}_files.txt')
        with open(output_path, 'w') as output_file:
            output_file.write('\n'.join(below_threshold_files))
        print(f"File paths saved to {output_path}")
    else:
        print(f"No files with {threshold} or fewer lines found.")

directory_path = 'AVG_ds/80/txt'
threshold_value = 24

save_below_n_files(directory_path, threshold_value)

