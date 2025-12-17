import numpy as np
import torch

def calculate_confusion_matrix(logits, labels, num_classes):
    # logits: モデルの出力 (batch_size, num_classes)
    # labels: 正解ラベル (batch_size)
    # num_classes: クラス数

    preds = torch.argmax(logits, dim=1)
    confusion_matrix = torch.zeros(num_classes, num_classes)

    for t, p in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def analyze_flower_pot_confusion(confusion_matrix, class_names, target_class='flower_pot'):
    target_index = class_names.index(target_class)
    actual_counts = confusion_matrix[target_index]
    predicted_counts = confusion_matrix[:, target_index]
    
    print(f"Actual class 'flower_pot' was predicted as:")
    for i, count in enumerate(actual_counts):
        print(f"  {class_names[i]}: {count} times")
    
    print(f"\nPredicted class 'flower_pot' when actual class was:")
    for i, count in enumerate(predicted_counts):
        print(f"  {class_names[i]}: {count} times")
