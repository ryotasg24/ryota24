import os

def count_files_in_directory(directory):
    """
    指定されたディレクトリを読み込み、ディレクトリ内のファイル数を返す関数。
    """
    try:
        file_count = len([f for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))])
        print(f"{directory} 内のファイル数: {file_count}")
        return file_count
    except Exception as e:
        print(f"エラー: {e}")
        return 0

# 使用例
target_directory = "/workspace/PointNeXt/result/dsSHAP_PointNeXt_h5/ply_files/50"
count_files_in_directory(target_directory)

