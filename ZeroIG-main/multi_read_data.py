import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task, target_dir=None):
        self.target_dir = target_dir
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        # 收集图像路径
        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.train_low_data_names.append(os.path.join(root, name))
        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        # 定义图像转换
        if self.task == 'train':
            # 训练时：添加颜色增强及其他数据增强
            # 注意：为了保持一致性，对输入和目标应用相同的随机变换
            self.transform = transforms.Compose([
                transforms.Resize((300, 200)),
                transforms.RandomHorizontalFlip(p=0.5),

                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                ),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.RandomGrayscale(p=0.01),
                transforms.ToTensor()

            ])
        else:
            # 测试时保持不变（不做增强，只转Tensor）
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # 记录数据集统计信息
        logging.info(f"Dataset initialized - Task: {task}, Low images: {self.count}")
        if target_dir:
            logging.info(f"Target directory: {target_dir}")

    def load_images_transform(self, file):
        im = Image.open(file).convert('RGB')
        im = self.transform(im)
        im = torch.clamp(im, 0.0, 1.0)  # 确保数据在[0,1]范围内
        return im


    def __getitem__(self, index):
        img_path = self.train_low_data_names[index]
        img_name = os.path.basename(img_path)
        low_img = self.load_images_transform(img_path)

        # 记录图像加载信息
        logging.debug(f"Loading image: {img_path}, Shape: {low_img.shape}")

        # 添加目标图像加载逻辑
        target_img = None
        if self.task == 'test' and self.target_dir:
            # 构建目标图像路径
            target_path = os.path.join(self.target_dir, img_name)
            if os.path.exists(target_path):
                target_img = self.load_images_transform(target_path)
                logging.debug(f"Loading target image: {target_path}, Shape: {target_img.shape}")
            else:
                # 如果找不到对应图像，创建全黑占位符并记录警告
                logger.warning(f"Target image not found: {target_path}. Using zero tensor.")
                target_img = torch.zeros_like(low_img)

        if self.task == 'train' and self.target_dir:
            # 训练模式：使用相对路径构建目标图像路径
            # 注意：这里假设低光图像和目标图像在相同目录结构下
            rel_path = os.path.relpath(img_path, self.low_img_dir)
            target_path = os.path.join(self.target_dir, rel_path)
            if os.path.exists(target_path):
                target_img = self.load_images_transform(target_path)
                logging.debug(f"Loading target image: {target_path}, Shape: {target_img.shape}")
            else:
                logger.warning(f"Target image not found: {target_path}. Using zero tensor.")
                target_img = torch.zeros_like(low_img)
            return low_img, target_img, img_name
        elif self.task == 'test' and self.target_dir:
            # 测试模式返回目标图像
            return low_img, target_img, img_name
        else:
            # 如果没有提供目标目录，则只返回低光图像和名称
            return low_img, img_name

    def __len__(self):
        return self.count