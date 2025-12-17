import os

def process_directories(root_directory, output_file):
    file_list = []
    with open(output_file, 'w') as output:
        for subdir, dirs, files in os.walk(root_directory):
            for file in files:
                if file.endswith(".txt"):
                    parts = file.split('_')
                    if len(parts) >= 2:
                        class_name, file_number = '_'.join(parts[:-1]), parts[-1]
                        file_number = file_number.split('.')[0].zfill(4)  # Remove file extension and right-pad with zeros
                        output_line = f"{class_name}_{file_number}"
                        file_list.append(output_line)

        for line in sorted(file_list):
            output.write(line + '\n')

root_directory = 'loss10_SampleNet50/eval/class_data'
output_file = 'finish_to_move/modelnet40_test.txt'

process_directories(root_directory, output_file)

