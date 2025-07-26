import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import os

# 自定义数据集类，继承PyTorch的Dataset，用于加载图像数据
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):
        # 初始化函数：读取图像目录并准备数据列表
        self.low_img_dir = img_dir  # 低光图像目录路径
        self.task = task  # 任务类型（'train'或'test'）
        self.train_low_data_names = []  # 存储低光图像路径的列表
        self.train_target_data_names = []  # 存储目标图像路径的列表（未使用，预留）

        # 遍历目录下所有文件，收集图像路径
        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))  # 拼接完整路径并添加到列表

        self.train_low_data_names.sort()  # 排序路径列表，确保顺序一致
        self.count = len(self.train_low_data_names)  # 数据总数

        # 定义图像转换：先缩放再转为Tensor（保持3:2比例，从2952x1728缩放到1476x864，缩小为原来的1/2）
        transform_list = []
        # 缩放图像（保持3:2比例），参数为(高度, 宽度)
        transform_list += [transforms.Resize((864, 1476))]
        transform_list += [transforms.ToTensor()]  # 将PIL图像转为Tensor（会自动归一化到[0,1]）
        self.transform = transforms.Compose(transform_list)  # 组合转换操作


    def load_images_transform(self, file):
        # 加载图像并应用转换
        im = Image.open(file).convert('RGB')  # 打开图像并转为RGB格式
        img_norm = self.transform(im).numpy()  # 应用转换并转为numpy数组
        img_norm = np.transpose(img_norm, (1, 2, 0))  # 调整维度顺序：(C,H,W)→(H,W,C)，便于后续处理
        return img_norm


    def __getitem__(self, index):
        # 按索引获取数据（PyTorch Dataset核心方法）
        # 加载低光图像
        low = self.load_images_transform(self.train_low_data_names[index])
        low = np.asarray(low, dtype=np.float32)  # 转为float32类型
        low = np.transpose(low[:, :, :], (2, 0, 1))  # 调整维度顺序为(C,H,W)，符合PyTorch张量格式
        # 获取图像文件名（用于保存结果时命名）
        img_name = self.train_low_data_names[index].split('\\')[-1]  # 分割路径，取最后一个部分（文件名）

        return torch.from_numpy(low), img_name  # 返回图像张量和文件名


    def __len__(self):
        # 返回数据总数（PyTorch Dataset核心方法）
        return self.count