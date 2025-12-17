import os

def count_files_in_directory(directory):
    file_count = 0

    for root, dirs, files in os.walk(directory):
        file_count += len(files)

    return file_count

if __name__ == '__main__':
    target_directory = "/code/ds_normal/data/modelnet40/FPS_ds/1000/ply"

    total_files = count_files_in_directory(target_directory)
    print(f"Total number of files in '{target_directory}': {total_files}")

