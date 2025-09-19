import json
import os
import random
from copy import deepcopy
from math import ceil

# 加载JSON文件
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 保存JSON文件并输出信息
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"{file_path}: {len(data['images'])} images, {len(data['annotations'])} annotations")

# 根据标注将图像分为有标注和无标注
def separate_images(data):
    annotated_images = set(anno["image_id"] for anno in data["annotations"])
    images_with_annotations = [img for img in data["images"] if img["id"] in annotated_images]
    return images_with_annotations

# 分割数据为k份，均匀分配剩余图像
def k_fold_split(images, annotations, k=5):
    random.shuffle(images)
    folds = [[] for _ in range(k)]
    for i, img in enumerate(images):
        fold_index = i % k
        folds[fold_index].append(img)
    
    annotation_folds = []
    for fold in folds:
        image_ids = set(img["id"] for img in fold)
        annotation_fold = [anno for anno in annotations if anno["image_id"] in image_ids]
        annotation_folds.append(annotation_fold)
    
    return folds, annotation_folds

# 创建k个目录，每个目录下有三个JSON文件
def create_k_folds(data, k=5):
    images_with_annotations = separate_images(data)
    
    if len(images_with_annotations) < k:
        print(f"Warning: Number of annotated images ({len(images_with_annotations)}) is less than k ({k}). Adjusting k to {len(images_with_annotations)}.")
        k = len(images_with_annotations)
    
    annotated_image_folds, annotated_annotation_folds = k_fold_split(images_with_annotations, data["annotations"], k)
    
    # 创建k个目录
    for i in range(k):
        fold_dir = f'annotations/fold_{i+1}'
        os.makedirs(fold_dir, exist_ok=True)

        # 训练集：排除当前折和下一个折
        train_images = []
        train_annotations = []
        for j in range(k):
            if j != i and j != (i + 1) % k:
                train_images.extend(annotated_image_folds[j])
                train_annotations.extend(annotated_annotation_folds[j])
        
        train_data = {
            "images": train_images,
            "annotations": train_annotations,
            "categories": deepcopy(data["categories"])
        }
        save_json(train_data, f'{fold_dir}/spine_2d_train.json')

        # 验证集：当前折
        val_images = annotated_image_folds[i]
        val_annotations = annotated_annotation_folds[i]
        val_data = {
            "images": val_images,
            "annotations": val_annotations,
            "categories": deepcopy(data["categories"])
        }
        save_json(val_data, f'{fold_dir}/spine_2d_val.json')

        # 测试集：下一个折
        test_fold = (i + 1) % k
        test_images = annotated_image_folds[test_fold]
        test_annotations = annotated_annotation_folds[test_fold]
        test_data = {
            "images": test_images,
            "annotations": test_annotations,
            "categories": deepcopy(data["categories"])
        }
        save_json(test_data, f'{fold_dir}/spine_2d_test.json')

# 主程序
if __name__ == "__main__":
    # 加载所有数据
    file_path = 'all_data.json'  # 确保这是正确的路径
    data = load_json(file_path)
    
    # 创建k折交叉验证数据
    create_k_folds(data, k=5)
