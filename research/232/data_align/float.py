import numpy as np
def detect_non_float_lines(file_path):
    non_float_lines = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                # 行を浮動小数点数に変換してみる
                _ = np.fromstring(line, dtype=float, sep=',')
            except ValueError:
                # 変換できない場合はエラーとして記録
                non_float_lines.append(line_number)

    return non_float_lines

# エラーが発生したファイルのパス
problematic_file_path = 'modelnet40/AVG_ds/5000/txt1000/curtain/curtain_0102.txt'

# 浮動小数点に変換できない行を検出
non_float_lines = detect_non_float_lines(problematic_file_path)

# 結果を表示
if non_float_lines:
    print(f"浮動小数点に変換できない行が見つかりました。行番号: {non_float_lines}")
else:
    print("すべての行が浮動小数点に変換可能です。")

