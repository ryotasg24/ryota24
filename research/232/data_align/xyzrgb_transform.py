import os

def convert_xyzrgb_to_xyz_recursive(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each directory in the input folder
    for dirpath, dirnames, filenames in os.walk(input_folder):
        # Create corresponding subdirectories in the output folder
        rel_path = os.path.relpath(dirpath, input_folder)
        output_subfolder = os.path.join(output_folder, rel_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Iterate through each file in the current directory
        for filename in filenames:
            if filename.endswith(".txt"):
                input_filepath = os.path.join(dirpath, filename)

                # Create the output filename with "_r" added
                output_filename = filename
                output_filepath = os.path.join(output_subfolder, output_filename)

                # Open the input file and read lines
                with open(input_filepath, 'r') as infile:
                    lines = infile.readlines()

                # Extract xyz coordinates from each line
                xyz_lines = [' '.join(line.strip().split(',')[:3]) + '\n' for line in lines]

                # Write the xyz lines to the output file
                with open(output_filepath, 'w') as outfile:
                    outfile.writelines(xyz_lines)

if __name__ == "__main__":
    input_folder = "data/gpcc/test"  # Replace with the path to your input folder
    output_folder = "data/gpcc/S40/txt"  # Replace with the path to your output folder

    convert_xyzrgb_to_xyz_recursive(input_folder, output_folder)
