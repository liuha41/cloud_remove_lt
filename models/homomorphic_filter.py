import os
import cv2
import numpy as np
import logging
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float
import sys


# 获取当前文件路径配置（保持不变）
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from util.niqe import niqe


def homofilter(img):
    I = np.float32(img)
    I3 = I
    m, n, chanels = I.shape
    rL = 0.5
    rH = 2
    c = 2
    d0 = 20
    for chanel in range(chanels):
        I1 = np.log(I[:, :, chanel] + 1)
        FI = np.fft.fft2(I1)
        n1 = np.floor(m / 2)
        n2 = np.floor(n / 2)
        D = np.zeros((m, n))
        H = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                D[i, j] = ((i - n1) **2 + (j - n2)** 2)
                H[i, j] = (rH - rL) * (np.exp(c * (-D[i, j] **2 / (d0** 2)))) + rL
        I2 = np.fft.ifft2(H * FI)
        I3[:, :, chanel] = np.real(np.exp(I2))
    max_index = np.unravel_index(I3.argmax(), I3.shape)
    maxV = I3[max_index[0], max_index[1], max_index[2]]
    min_index = np.unravel_index(I3.argmin(), I3.shape)
    minV = I3[min_index[0], min_index[1], min_index[2]]
    for chanel in range(chanels):
        for i in range(m):
            for j in range(n):
                I3[i, j, chanel] = 255 * (I3[i, j, chanel] - minV) / (maxV - minV)
    return np.uint8(I3)


def process_images(cloud_dir, label_dir, save_dir):
    # 1. 修改文件筛选条件：加入 .tif 和 .tiff 后缀
    cloud_files = [f for f in os.listdir(cloud_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]  # 新增 tif 格式
    total_psnr, total_ssim, total_niqe = 0.0, 0.0, 0.0
    count = 0

    for file in cloud_files:
        cloud_path = os.path.join(cloud_dir, file)
        label_path = os.path.join(label_dir, file)

        if not os.path.exists(label_path):
            logging.warning(f"清晰图不存在：{label_path}，已跳过")
            continue

        # 2. 读取图像（OpenCV 原生支持 tif 格式，无需额外修改）
        cloud_img = cv2.imread(cloud_path)
        label_img = cv2.imread(label_path)

        # 3. 特殊处理：若 tif 是单通道灰度图，转为三通道（避免后续处理报错）
        if cloud_img is not None and len(cloud_img.shape) == 2:
            cloud_img = cv2.cvtColor(cloud_img, cv2.COLOR_GRAY2BGR)
        if label_img is not None and len(label_img.shape) == 2:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)

        if cloud_img is None or label_img is None:
            logging.warning(f"图像读取失败（可能是不支持的 tif 格式）：{file}，已跳过")
            continue

        if cloud_img.shape != label_img.shape:
            label_img = cv2.resize(label_img, (cloud_img.shape[1], cloud_img.shape[0]))

        filtered_img = homofilter(cloud_img)

        # 4. 保存结果（支持 tif 格式，OpenCV 会根据文件名后缀自动处理）
        save_path = os.path.join(save_dir, file)
        cv2.imwrite(save_path, filtered_img)

        # 后续指标计算逻辑（保持不变）
        filtered_gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        label_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

        filtered_float = img_as_float(filtered_gray)
        label_float = img_as_float(label_gray)

        current_psnr = psnr(label_float, filtered_float, data_range=1.0)
        current_ssim = ssim(label_gray, filtered_gray, data_range=filtered_gray.max() - filtered_gray.min())
        current_niqe = niqe(filtered_gray)

        total_psnr += current_psnr
        total_ssim += current_ssim
        total_niqe += current_niqe
        count += 1

        logging.info(
            f"处理完成：{file} | "
            f"PSNR: {current_psnr:.2f} | "
            f"SSIM: {current_ssim:.4f} | "
            f"NIQE: {current_niqe:.2f}"
        )

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_niqe = total_niqe / count
        logging.info("\n===== 平均指标 =====")
        logging.info(f"平均PSNR: {avg_psnr:.2f}")
        logging.info(f"平均SSIM: {avg_ssim:.4f}")
        logging.info(f"平均NIQE: {avg_niqe:.2f}")
    else:
        logging.info("未处理任何有效图像")


if __name__ == "__main__":
    # 命令行参数检查（保持与之前一致，支持 dataset 名称传入）
    if len(sys.argv) != 5:
        print("参数错误！请按格式传入：")
        print("python homomorphic_filter.py <dataset名称> <云雾图文件夹> <清晰图文件夹> <结果保存文件夹>")
        sys.exit(1)

    dataset = sys.argv[1]
    cloud_dir = sys.argv[2]
    label_dir = sys.argv[3]
    save_dir = sys.argv[4]

    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.dirname(save_dir)
    log_file = os.path.join(log_dir, f"{dataset}_homomorphic_filter.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    process_images(cloud_dir, label_dir, save_dir)