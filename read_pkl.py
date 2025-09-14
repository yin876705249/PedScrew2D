import pickle

# mmpose 生成的 pkl 文件路径
pkl_file_path = 'ViTPose_fold_1.pkl'

with open(pkl_file_path, 'rb') as file:
    keypoints_data = pickle.load(file)

# 查看数据结构
print(keypoints_data)

# 示例：遍历数据
for image_name, persons in keypoints_data.items():
    print(f"Image: {image_name}")
    for idx, person in enumerate(persons):
        print(f"  Person {idx + 1}:")
        for kp in person:
            print(f"    x: {kp['x']}, y: {kp['y']}, v: {kp['v']}")