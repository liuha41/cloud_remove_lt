import os
import requests
import rarfile


def main(url, dataset):
    # download
    print(f'Download {url} ...')
    response = requests.get(url)
    with open(f'datasets/{dataset}.zip', 'wb') as f:
        f.write(response.content)

    # unzip
    import zipfile
    with zipfile.ZipFile(f'datasets/{dataset}.zip', 'r') as zip_ref:
        zip_ref.extractall(f'datasets/')


if __name__ == '__main__':

    urls = [
        'https://cloud-removal-dataset.obs.cn-north-4.myhuaweicloud.com/whu_cloud_dataset.zip'  # 替换为实际URL
    ]
    datasets = ['whu_cloud_dataset']

    for url, dataset in zip(urls, datasets):
        main(url, dataset)
