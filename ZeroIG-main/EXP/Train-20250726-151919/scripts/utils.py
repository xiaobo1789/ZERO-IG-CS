import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image


def pair_downsampler(img):
    # 对图像进行双路下采样（用于多尺度处理）
    # img形状：B C H W（批次、通道、高、宽）
    c = img.shape[1]  # 通道数
    # 定义两个下采样滤波器（2x2）
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)  # 滤波器1：对角线采样
    filter1 = filter1.repeat(c, 1, 1, 1)  # 按通道复制（与输入通道数匹配）
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)  # 滤波器2：反对角线采样
    filter2 = filter2.repeat(c, 1, 1, 1)
    # 卷积下采样（步长2，分组卷积=通道数，相当于每个通道独立处理）
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2  # 返回两个下采样结果


def gauss_cdf(x):
    # 高斯分布的累积分布函数（CDF）
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))  # 利用误差函数erf计算


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    # 生成高斯核（用于模糊处理）
    interval = (2 * nsig + 1.) / (kernlen)  # 计算区间步长
    # 生成高斯分布的x坐标
    x = torch.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1,).cuda()
    kern1d = torch.diff(gauss_cdf(x))  # 1D高斯核（通过CDF差分得到）
    kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))  # 2D高斯核（外积开方）
    kernel = kernel_raw / torch.sum(kernel_raw)  # 归一化
    out_filter = kernel.view(1, 1, kernlen, kernlen)  # 调整形状为(1,1,kernlen,kernlen)
    out_filter = out_filter.repeat(channels, 1, 1, 1)  # 按通道复制
    return out_filter


class LocalMean(torch.nn.Module):
    # 计算图像的局部均值（用于纹理分析）
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size  # 局部窗口大小
        self.padding = self.patch_size // 2  # 填充大小（保持输出尺寸）

    def forward(self, image):
        # 填充图像（反射模式，避免边界效应）
        image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        # 提取滑动窗口 patches：形状(B,C,H',W',patch_size,patch_size)
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))  # 计算每个窗口的均值（沿patch的高和宽）


def blur(x):
    # 对图像进行高斯模糊
    device = x.device  # 获取输入设备
    kernel_size = 21  # 高斯核大小
    padding = kernel_size // 2  # 填充大小
    # 生成高斯核（与输入通道数匹配）
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(device)
    # 填充图像
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    # 卷积模糊（分组卷积，每个通道独立处理）
    return torch.nn.functional.conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))


def padr_tensor(img):
    # 对张量进行边缘填充（填充2个像素，值为0）
    pad = 2
    pad_mod = torch.nn.ConstantPad2d(pad, 0)  # 常量填充（值为0）
    img_pad = pad_mod(img)
    return img_pad


def calculate_local_variance(train_noisy):
    # 计算图像的局部方差（用于噪声估计）
    b, c, w, h = train_noisy.shape  # 批次、通道、宽、高
    avg_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)  # 平均池化（计算局部均值）
    noisy_avg = avg_pool(train_noisy)  # 局部均值
    noisy_avg_pad = padr_tensor(noisy_avg)  # 填充均值图
    train_noisy = padr_tensor(train_noisy)  # 填充原图
    # 提取滑动窗口（5x5）
    unfolded_noisy_avg = noisy_avg_pad.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy = train_noisy.unfold(2, 5, 1).unfold(3, 5, 1)
    # 调整形状以便计算方差
    unfolded_noisy_avg = unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0], -1, 5, 5)
    unfolded_noisy = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    # 计算每个窗口的方差（(x-mean)^2的均值）
    noisy_diff_squared = (unfolded_noisy - unfolded_noisy_avg) ** 2
    noisy_var = torch.mean(noisy_diff_squared, dim=(2, 3))
    noisy_var = noisy_var.view(b, c, w, h)  # 恢复原始形状
    return noisy_var


def count_parameters_in_MB(model):
    # 计算模型参数总量（单位：MB）
    # 遍历所有参数，计算总元素数，除以1e6（1MB=1e6元素）
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    # 保存模型检查点（含优化器状态等）
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:  # 如果是最佳模型，额外保存一份
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    # 仅保存模型权重
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    # 路径丢弃（用于随机深度网络，防止过拟合）
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # 生成掩码（伯努利分布，保留概率keep_prob）
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)  # 除以保留概率（保持期望不变）
        x.mul_(mask)  # 应用掩码
    return x


def create_exp_dir(path, scripts_to_save=None):
    # 创建实验目录，并保存相关脚本（用于复现）
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('实验目录 : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)  # 复制脚本到实验目录


def show_pic(pic, name, path):
    # 显示并保存图像（用于可视化中间结果）
    pic_num = len(pic)  # 图像数量
    for i in range(pic_num):
        img = pic[i]
        image_numpy = img[0].cpu().float().numpy()  # 转为numpy数组
        if image_numpy.shape[0] == 3:  # RGB图像
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))  # 调整维度
            im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)  # 子图位置
            plt.xlabel(str(img_name))  # 标题
            plt.xticks([])  # 隐藏坐标轴
            plt.yticks([])
            plt.imshow(im)
        elif image_numpy.shape[0] == 1:  # 灰度图像
            im = Image.fromarray(np.clip(image_numpy[0] * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im, plt.cm.gray)  # 灰度显示
    plt.savefig(path)  # 保存图像