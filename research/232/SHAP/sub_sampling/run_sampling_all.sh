## for PointNeXt
#!/bin/bash

 サンプリング手法のリスト
for method in AVG US; do
    # 指定点数のリスト
    for num in 50 75 100 250 500 750 1000; do
        echo "Running conventional_sampling.py with method ${method} and num_samples ${num}"
        CUDA_VISIBLE_DEVICES=0 python conventional_sampling.py --method ${method} --num_samples ${num}
    done
done