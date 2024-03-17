import os

def extract_file_paths(directory, output_file):
    with open(output_file, "w") as output:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, directory)
                    output.write(relative_path + "\n")


input_directory = "loss10_SampleNet50/eval/class_data" 
output_file = "finish_to_move/filelist.txt"

extract_file_paths(input_directory, output_file)

