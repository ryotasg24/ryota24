import os

def check_txt_file_line_count(directory_path, target_line_count):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    line_count = sum(1 for line in file)
                if line_count != target_line_count:
                    print(f"File {file_path} does not have {target_line_count} lines. Actual lines: {line_count}")

target_directory = '/code/ds_normal/data/gpcc/50/txt'
target_line_count = 50

check_txt_file_line_count(target_directory, target_line_count)

