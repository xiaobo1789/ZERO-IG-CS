import os
import numpy as np
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import logging  # 添加logging模块导入


def save(model, path):
    torch.save(model.state_dict(), path)


def count_parameters_in_MB(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def pair_downsampler(img):
    device = img.device
    dtype = img.dtype
    B, C, H, W = img.shape



    # 确保高度和宽度是偶数
    if H % 2 != 0:
        img = F.pad(img, (0, 0, 0, 1), mode='reflect')
        H += 1
    if W % 2 != 0:
        img = F.pad(img, (0, 1, 0, 0), mode='reflect')
        W += 1

    # 确保掩码在相同设备和数据类型上
    mask1 = torch.tensor([[[[1, 0], [0, 1]]]], dtype=dtype, device=device)
    mask2 = torch.tensor([[[[0, 1], [1, 0]]]], dtype=dtype, device=device)

    mask1 = mask1.repeat(C, 1, 1, 1)
    mask2 = mask2.repeat(C, 1, 1, 1)

    output1 = F.conv2d(img, mask1, stride=2, groups=C) * 2
    output2 = F.conv2d(img, mask2, stride=2, groups=C) * 2


    return output1, output2


def gauss_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))


# +++ 添加：更真实的微光图像退化函数 +++
def degrade_image(image, severity):
    """
    根据严重程度应用一系列真实的微光图像退化。
    severity: 0.0 (无退化) 到 1.0 (严重退化) 的浮点数。
    """
    if severity == 0:
        return image

    device = image.device

    # 1. 模拟欠曝光 (Gamma校正)
    # 严重程度越高，gamma值越大，图像越暗
    gamma = 1.0 + severity * 2.0
    image_darkened = image ** gamma

    # 2. 模拟色彩偏移 (Color Cast)
    # 随机选择一个颜色通道进行增强或减弱
    color_cast = torch.tensor([1.0, 1.0, 1.0], device=device)
    channel_to_cast = np.random.randint(0, 3)
    cast_direction = np.random.choice([-1, 1])
    # 严重程度越高，色彩偏移越明显
    cast_amount = 0.1 + severity * 0.3
    color_cast[channel_to_cast] += cast_direction * cast_amount
    image_color_cast = image_darkened * color_cast.view(1, 3, 1, 1)

    # 3. 模拟传感器噪声 (泊松-高斯噪声模型)
    # 泊松分量 (与信号强度相关)
    shot_noise_strength = 0.1 * severity
    image_with_shot_noise = torch.poisson(image_color_cast / shot_noise_strength) * shot_noise_strength

    # 高斯分量 (读出噪声，与信号无关)
    read_noise_strength = 0.05 * severity
    read_noise = torch.randn_like(image_with_shot_noise) * read_noise_strength

    image_noisy = image_with_shot_noise + read_noise

    return torch.clamp(image_noisy, 0, 1)
def gauss_kernel(kernlen=21, nsig=3, channels=1, device='cpu'):
    interval = (2 * nsig + 1.) / kernlen
    x = torch.linspace(-nsig - interval / 2., nsig + interval / 2., steps=kernlen + 1, device=device)
    kern1d = torch.diff(0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2., device=device)))))
    kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
    kernel = kernel_raw / torch.sum(kernel_raw)
    out_filter = kernel.view(1, 1, kernlen, kernlen).repeat(channels, 1, 1, 1)
    return out_filter


class LocalMean(torch.nn.Module):
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, image):
        image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))


def blur(x):
    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1), device=device)
    x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))


def padr_tensor(img):
    pad = 2
    pad_mod = torch.nn.ConstantPad2d(pad, 0)
    img_pad = pad_mod(img)
    return img_pad


def calculate_local_variance(train_noisy):
    b, c, w, h = train_noisy.shape
    avg_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    noisy_avg = avg_pool(train_noisy)
    noisy_avg_pad = padr_tensor(noisy_avg)
    train_noisy = padr_tensor(train_noisy)
    unfolded_noisy_avg = noisy_avg_pad.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy = train_noisy.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy_avg = unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0], -1, 5, 5)
    unfolded_noisy = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    noisy_diff_squared = (unfolded_noisy - unfolded_noisy_avg) ** 2
    noisy_var = torch.mean(noisy_diff_squared, dim=(2, 3))
    noisy_var = noisy_var.view(b, c, w, h)

    # 添加方差统计
    var_stats = {
        'min': noisy_var.min().item(),
        'max': noisy_var.max().item(),
        'mean': noisy_var.mean().item(),
        'std': noisy_var.std().item()
    }
    logging.debug(f"Local variance stats: {var_stats}")

    return noisy_var


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('实验目录 : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def show_pic(pic, name, path):
    pic_num = len(pic)
    for i in range(pic_num):
        img = pic[i]
        image_numpy = img[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im)
        elif image_numpy.shape[0] == 1:
            im = Image.fromarray(np.clip(image_numpy[0] * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im, plt.cm.gray)
    plt.savefig(path)