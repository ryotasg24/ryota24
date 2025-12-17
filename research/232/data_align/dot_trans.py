import os

input_directory = 'AVG_ds/2500/txt'

for subdir, _, files in os.walk(input_directory):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(subdir, file)
            output_path = os.path.join(f'AVG_ds/5000/txt2500/{file}')

            with open(file_path, 'r') as input_file, open(output_path, 'w') as output_file:
                for line in input_file:
                    new_line = ','.join(line.split())
                    output_file.write(new_line + '\n')

print("Reshaping completed.")
