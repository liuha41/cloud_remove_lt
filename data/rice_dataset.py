import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class RICEDataset(data.Dataset):
    """RICE 云雾去除数据集"""

    def __init__(self, opt):
        super(RICEDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # 修改：分别指定清晰图片和云雾图片的路径
        self.gt_folder = opt['dataroot_gt']  # 清晰图片（无云雾）
        self.lq_folder = opt['dataroot_lq']  # 云雾遮挡图片

        # 获取对应的文件对
        self.gt_paths = [os.path.join(self.gt_folder, v) for v in list(scandir(self.gt_folder))]
        self.lq_paths = [os.path.join(self.lq_folder, v) for v in list(scandir(self.lq_folder))]

        # 确保文件对应（根据文件名匹配）
        assert len(self.gt_paths) == len(self.lq_paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # 加载清晰图片 (GT)
        gt_path = self.gt_paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # 加载云雾图片 (LQ) - 修改点：直接读取云雾图像而不是生成
        lq_path = self.lq_paths[index]
        lq_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(lq_bytes, float32=True)

        # 数据增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # 随机裁剪
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale=1, gt_path=gt_path)
            # 翻转旋转
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # 格式转换
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 标准化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.gt_paths)