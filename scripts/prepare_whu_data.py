import os
import requests
import rarfile


def main(url, dataset):
    # download
    print(f'Download {url} ...')
    response = requests.get(url)
    with open(f'datasets/whu_cloud_dataset/{dataset}.rar', 'wb') as f:
        f.write(response.content)

    with rarfile.RarFile(f'datasets/whu_cloud_dataset/{dataset}.rar', 'r') as rar_ref:
        rar_ref.extractall(f'datasets/whu_cloud_dataset/{dataset}')


if __name__ == '__main__':
    os.makedirs('datasets/whu_cloud_dataset', exist_ok=True)

    urls = [
        'https://cloud-removal-dataset.obs.cn-north-4.myhuaweicloud.com/cloud%20detection%20and%20removal%20dataset.rar'  # 替换为实际URL
    ]
    datasets = ['whu_cloud_dataset']

    for url, dataset in zip(urls, datasets):
        main(url, dataset)
