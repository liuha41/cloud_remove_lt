import os
import requests
import zipfile
import random
import shutil


def main(url, dataset):
    # download
    print(f'Download {url} ...')
    response = requests.get(url)
    with open(f'../datasets/rice/{dataset}.zip', 'wb') as f:
        f.write(response.content)

    # unzip
    with zipfile.ZipFile(f'../datasets/rice/{dataset}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'datasets/rice/{dataset}')

    # split datasets
    split_rice_datasets()


def split_rice_datasets():
    """为RICE1和RICE2数据集按8:2比例分割训练集和测试集"""

    for rice_dataset in ['RICE1', 'RICE2']:
        base_path = f'../datasets/rice/RICE_DATASET/{rice_dataset}'
        cloud_path = os.path.join(base_path, 'cloud')
        label_path = os.path.join(base_path, 'label')

        if not os.path.exists(cloud_path) or not os.path.exists(label_path):
            continue

        # 创建输出目录
        train_cloud_path = f'../datasets/rice/{rice_dataset}_train/cloud'
        train_label_path = f'../datasets/rice/{rice_dataset}_train/label'
        test_cloud_path = f'../datasets/rice/{rice_dataset}_test/cloud'
        test_label_path = f'../datasets/rice/{rice_dataset}_test/label'

        for path in [train_cloud_path, train_label_path, test_cloud_path, test_label_path]:
            os.makedirs(path, exist_ok=True)

        # 获取文件列表
        cloud_files = sorted([f for f in os.listdir(cloud_path)
                              if os.path.isfile(os.path.join(cloud_path, f))])

        # 创建文件对
        file_pairs = []
        for cloud_file in cloud_files:
            # cloud_name = os.path.splitext(cloud_file)[0]
            label_file = cloud_file  # 假设同名
            if os.path.exists(os.path.join(label_path, label_file)):
                file_pairs.append((cloud_file, label_file))

        # 随机分割
        random.shuffle(file_pairs)
        split_idx = int(0.8 * len(file_pairs))
        train_pairs = file_pairs[:split_idx]
        test_pairs = file_pairs[split_idx:]

        # 复制文件
        for cloud_file, label_file in train_pairs:
            shutil.copy2(os.path.join(cloud_path, cloud_file),
                         os.path.join(train_cloud_path, cloud_file))
            shutil.copy2(os.path.join(label_path, label_file),
                         os.path.join(train_label_path, label_file))

        for cloud_file, label_file in test_pairs:
            shutil.copy2(os.path.join(cloud_path, cloud_file),
                         os.path.join(test_cloud_path, cloud_file))
            shutil.copy2(os.path.join(label_path, label_file),
                         os.path.join(test_label_path, label_file))

        # 处理RICE2的mask
        if rice_dataset == 'RICE2':
            mask_path = os.path.join(base_path, 'mask')
            if os.path.exists(mask_path):
                train_mask_path = f'../datasets/rice/{rice_dataset}_train/mask'
                test_mask_path = f'../datasets/rice/{rice_dataset}_test/mask'
                os.makedirs(train_mask_path, exist_ok=True)
                os.makedirs(test_mask_path, exist_ok=True)

                for cloud_file, _ in train_pairs:
                    mask_file = cloud_file
                    shutil.copy2(os.path.join(mask_path, mask_file),
                                 os.path.join(train_mask_path, mask_file))

                for cloud_file, _ in test_pairs:
                    mask_file = cloud_file
                    shutil.copy2(os.path.join(mask_path, mask_file),
                                 os.path.join(test_mask_path, mask_file))


if __name__ == '__main__':
    os.makedirs('../datasets/rice', exist_ok=True)

    urls = [
        'https://cloud-removal-dataset.obs.cn-north-4.myhuaweicloud.com/RICE_DATASET.zip'  # 替换为实际URL
    ]
    datasets = ['RICE_DATASET']

    for url, dataset in zip(urls, datasets):
        main(url, dataset)
