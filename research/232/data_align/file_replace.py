import os
import shutil

def replace_files(source_directory, target_directory, filenames_to_replace):
    for root, _, files in os.walk(source_directory):
        for filename in files:
            if filename in filenames_to_replace:
                source_file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(source_file_path, source_directory)
                target_file_path = os.path.join(target_directory, relative_path)

                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)

                shutil.copyfile(source_file_path, target_file_path)
                print(f"File replaced: {filename}")

# from_directory
source_directory = '../raw_modelnet40'

# to_directory
target_directory = 'AVG_ds/500/txt'

# file_name
filenames_to_replace = ['airplane_0266.txt','airplane_0617.txt','airplane_0299.txt','airplane_0569.txt','tv_stand_0012.txt','tv_stand_0055.txt','keyboard_0088.txt','stairs_0012.txt','stairs_0079.txt','guitar_0161.txt','guitar_0207.txt','guitar_0191.txt','guitar_0109.txt','guitar_0045.txt','xbox_0123.txt','plant_0071.txt','curtain_0071.txt','vase_0354.txt','flower_pot_0113.txt', 'radio_0019.txt','radio_0105.txt','bottle_0064.txt','bottle_0106.txt','bottle_0036.txt','bottle_0154.txt','bottle_0296.txt','cup_0058.txt']


replace_files(source_directory, target_directory, filenames_to_replace)
