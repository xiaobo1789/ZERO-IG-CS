import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import os

import matplotlib.pyplot as plt
import sys
import time
import torch
import math
import glob
import numpy as np
import utils
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from PIL import Image
import logging
import argparse
from torch.utils.data import DataLoader as TorchDataLoader
from model import Network
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *
from multi_read_data import DataLoader
import gc
import pyiqa
import lpips as lpips_lib
from adamp import AdamP
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置CUDA环境和优化选项
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

# 解析命令行参数
parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')

parser.add_argument('--cuda', type=bool, default=True, help='是否使用CUDA训练')
parser.add_argument('--gpu', type=str, default='0', help='GPU设备ID')
parser.add_argument('--seed', type=int, default=2, help='随机种子')
parser.add_argument('--epochs', type=int, default=6001, help='训练轮数')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='生成器学习率')
parser.add_argument('--lr_disc', type=float, default=1e-4, help='判别器学习率')
parser.add_argument('--save', type=str, default='./EXP/', help='实验结果保存根目录')
parser.add_argument('--model_pretrain', type=str, default='', help='预训练模型路径')
parser.add_argument('--adv_weight', type=float, default=0.08, help='对抗损失权重')
parser.add_argument('--patience', type=int, default=1000, help='早停耐心值')
parser.add_argument('--min_delta', type=float, default=0.002, help='最小提升阈值')
parser.add_argument('--disc_update_freq', type=int, default=2, help='判别器更新频率')
parser.add_argument('--gradient_penalty_weight', type=float, default=2.0, help='梯度惩罚权重')
args = parser.parse_args()

# 设置可见GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 设备数量: {torch.cuda.device_count()}")
print(f"当前 CUDA 设备: {torch.cuda.current_device()}")

# 创建实验目录
args.save = os.path.join(args.save, f"Train-{time.strftime('%Y%m%d-%H%M%S')}")
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = os.path.join(args.save, 'model_epochs/')
os.makedirs(model_path, exist_ok=True)
image_path = os.path.join(args.save, 'image_epochs/')
os.makedirs(image_path, exist_ok=True)

# 配置日志
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("train file name = %s", os.path.split(__file__)[1])

# 设置默认tensor类型
if torch.cuda.is_available() and args.cuda:
    torch.set_default_dtype(torch.float32)
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor):
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def print_memory_usage(stage):
    alloc = torch.cuda.memory_allocated() / 1024 ** 3  # GB
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"[{stage}] 已分配: {alloc:.2f}GB, 已预留: {reserved:.2f}GB")


def analyze_training_metrics(metrics_path):
    """分析训练指标"""
    try:
        df = pd.read_csv(metrics_path)
        if df.empty:
            logging.warning("Metrics CSV file is empty.")
            return None

        # 检查必要的列是否存在
        required_columns = ['Epoch', 'PSNR', 'SSIM', 'LPIPS', 'NIQE']
        for col in required_columns:
            if col not in df.columns:
                logging.warning(f"Column {col} not found in metrics file")
                return None

        # 处理NaN值 - 使用新方法
        df = df.ffill().bfill()

        # 确保所有指标列都是数值类型
        for col in ['PSNR', 'SSIM', 'LPIPS', 'NIQE']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 再次处理可能的NaN
        df = df.ffill().bfill()

        # 按epoch分组计算平均指标
        epoch_metrics = df.groupby('Epoch').agg({
            'PSNR': 'mean',
            'SSIM': 'mean',
            'LPIPS': 'mean',
            'NIQE': 'mean'
        }).reset_index()

        # 找到每个指标的最佳epoch
        best_psnr_idx = epoch_metrics['PSNR'].idxmax()
        best_ssim_idx = epoch_metrics['SSIM'].idxmax()
        best_lpips_idx = epoch_metrics['LPIPS'].idxmin()
        best_niqe_idx = epoch_metrics['NIQE'].idxmin()

        # 获取最佳指标值
        best_psnr = epoch_metrics.loc[best_psnr_idx, ['Epoch', 'PSNR']]
        best_ssim = epoch_metrics.loc[best_ssim_idx, ['Epoch', 'SSIM']]
        best_lpips = epoch_metrics.loc[best_lpips_idx, ['Epoch', 'LPIPS']]
        best_niqe = epoch_metrics.loc[best_niqe_idx, ['Epoch', 'NIQE']]

        # 计算综合得分
        epoch_metrics['norm_psnr'] = (epoch_metrics['PSNR'] - epoch_metrics['PSNR'].min()) / (
                epoch_metrics['PSNR'].max() - epoch_metrics['PSNR'].min() + 1e-8)
        epoch_metrics['norm_ssim'] = (epoch_metrics['SSIM'] - epoch_metrics['SSIM'].min()) / (
                epoch_metrics['SSIM'].max() - epoch_metrics['SSIM'].min() + 1e-8)
        epoch_metrics['norm_lpips'] = 1 - (epoch_metrics['LPIPS'] - epoch_metrics['LPIPS'].min()) / (
                epoch_metrics['LPIPS'].max() - epoch_metrics['LPIPS'].min() + 1e-8)
        epoch_metrics['norm_niqe'] = 1 - (epoch_metrics['NIQE'] - epoch_metrics['NIQE'].min()) / (
                epoch_metrics['NIQE'].max() - epoch_metrics['NIQE'].min() + 1e-8)

        epoch_metrics['composite_score'] = (
                epoch_metrics['norm_psnr'] * 0.4 +
                epoch_metrics['norm_ssim'] * 0.4 +
                epoch_metrics['norm_lpips'] * 0.1 +
                epoch_metrics['norm_niqe'] * 0.1
        )

        best_composite_idx = epoch_metrics['composite_score'].idxmax()
        best_composite = epoch_metrics.loc[best_composite_idx, ['Epoch', 'PSNR', 'SSIM', 'LPIPS', 'NIQE']]

        return {
            'best_psnr': best_psnr,
            'best_ssim': best_ssim,
            'best_lpips': best_lpips,
            'best_niqe': best_niqe,
            'best_composite': best_composite
        }
    except Exception as e:
        logging.error(f"分析训练指标时出错: {e}")
        return None


def write_best_metrics_to_log(metrics_path, log_path):
    """将最佳指标写入日志文件"""
    best_metrics = analyze_training_metrics(metrics_path)
    if not best_metrics:
        return

    with open(log_path, 'r') as f:
        original_content = f.read()

    best_info = f"""最佳指标总结:
PSNR最佳: epoch {int(best_metrics['best_psnr']['Epoch'])} - {best_metrics['best_psnr']['PSNR']:.4f}
SSIM最佳: epoch {int(best_metrics['best_ssim']['Epoch'])} - {best_metrics['best_ssim']['SSIM']:.4f}
LPIPS最佳: epoch {int(best_metrics['best_lpips']['Epoch'])} - {best_metrics['best_lpips']['LPIPS']:.4f}
NIQE最佳: epoch {int(best_metrics['best_niqe']['Epoch'])} - {best_metrics['best_niqe']['NIQE']:.4f}
综合最佳: epoch {int(best_metrics['best_composite']['Epoch'])} - PSNR: {best_metrics['best_composite']['PSNR']:.4f}, SSIM: {best_metrics['best_composite']['SSIM']:.4f}, LPIPS: {best_metrics['best_composite']['LPIPS']:.4f}, NIQE: {best_metrics['best_composite']['NIQE']:.4f}

"""

    with open(log_path, 'w') as f:
        f.write(best_info + original_content)

    logging.info(best_info)


def normalize_for_discriminator(x):
    """将输入图像裁剪到[0, 1]范围，匹配真实图像分布"""
    return torch.clamp(x, 0, 1)


# +++ 修改：实现零中心梯度惩罚 (0-GP) 以提升判别器泛化能力 +++
def compute_gradient_penalty(D, real_samples):
    """计算应用于真实样本的零中心梯度惩罚 (R1 正则化)"""
    real_samples.requires_grad_(True)
    d_real = D(real_samples)
    grad_outputs = torch.ones_like(d_real, requires_grad=False)

    gradients = torch.autograd.grad(
        outputs=d_real,
        inputs=real_samples,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )

    gradients = gradients.view(gradients.size(0), -1)
    # 惩罚梯度范数的平方，使其趋向于0
    gradient_penalty = (gradients.norm(2, dim=1) ** 2).mean()
    return gradient_penalty
    # +++ 修改：实现零中心梯度惩罚 (0-GP) 以提升判别器泛化能力 +++

class EMA:
    """指数移动平均"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class EarlyStopping:
    def __init__(self, patience=2000, min_delta=0.0005, warmup_epochs=1000):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, composite_score, current_epoch):
        if current_epoch < self.warmup_epochs:
            return False

        if self.best_score is None:
            self.best_score = composite_score
        elif composite_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = composite_score
            self.counter = 0
        return self.early_stop


def adjust_loss_weights(epoch):
    """动态调整损失权重"""
    # 前500epoch：主要学习基础重建
    if epoch < 500:
        weights = {
            'pixel_reconstruction': 1.5,
            'perceptual': 0.2,
            'texture_preserve': 0.3,
            'color_constancy': 0.1,
            'histogram_match': 0.1,
            'ms_ssim': 0.8,
            'frequency': 0.1,
            'noise_aware': 0.5,
            'psnr': 0.1,
            'overexposure_weight': 0.4,
            'adv_weight': 0.05  # 大幅降低对抗损失权重
        }
    # 500-2000epoch：平衡各项损失
    elif epoch < 1500:
        weights = {
            'pixel_reconstruction': 1.5 - 0.7 * (epoch - 500) / 1000,
            'perceptual': min(0.4, 0.2 + 0.6 * (epoch - 500) / 1000),
            'texture_preserve': min(0.5, 0.3 + 0.5 * (epoch - 500) / 1000),
            'color_constancy': 0.1,
            'histogram_match': 0.1 + 0.1 * (epoch - 500) / 1000,
            'ms_ssim': min(1.2, 0.8 + 0.4 * (epoch - 500) / 1000),
            'frequency': min(0.2, 0.1 + 0.1 * (epoch - 500) / 1000),
            'noise_aware': 0.5 + 0.3 * (epoch - 500) / 1000,
            'psnr': min(0.2, 0.1 + 0.1 * (epoch - 500) / 1500),
            'overexposure_weight': min(0.6, 0.3 + 0.3 * (epoch - 500) / 1000),
            'adv_weight': 0.05 + 0.1 * (epoch - 500) / 1000,
        }
    # 2000epoch后：专注于感知质量和细节
    else:
        weights = {
            'pixel_reconstruction': 0.8,
            'perceptual': 0.8,
            'texture_preserve': 0.8,
            'color_constancy': 0.1,
            'histogram_match': 0.2,
            'ms_ssim': 1.2,
            'frequency': 0.2,
            'noise_aware': 0.8,
            'psnr': 0.2,
            'overexposure_weight': 0.5,
            'adv_weight': 0.15
        }

    return weights




def adaptive_brightness_control(image, max_brightness=0.92, min_avg_brightness=0.4):
    """
    新增逻辑：
    - 若平均亮度 < 0.3（正常下限），按比例提升亮度
    - 过曝处理保留，但降低亮度衰减系数（从 0.9/0.8 改为 0.95/0.9）
    """
    # 计算图像平均亮度
    brightness = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
    avg_brightness = torch.mean(brightness)
    overexposed = (brightness > max_brightness).float()
    overexposed_ratio = overexposed.mean()

    # 1. 亮度不足时：按比例提升（目标达到 min_avg_brightness）
    if avg_brightness < min_avg_brightness:
        scale = min_avg_brightness / (avg_brightness + 1e-6)  # 提升比例（如 0.3/0.15=2.0）
        scale = torch.clamp(scale, 1.0, 3.0)
        image = image * scale

    # 2. 过曝时：轻微降低亮度（衰减系数从 0.9/0.8 改为 0.95/0.9，减少过度抑制）
    if overexposed_ratio > 0.1:
        image = image * 0.95  # 原 0.9 → 0.95
    elif overexposed_ratio > 0.05:
        mask = overexposed.unsqueeze(0).expand_as(image)
        image = torch.where(mask > 0, image * 0.9, image)  # 原 0.8 → 0.9

    return torch.clamp(image, 0, 1)


def check_nan_inf(tensor, name):
    """检查张量中是否有NaN或Inf值"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logging.warning(f"Warning: {name} contains NaN or Inf values.")
        return True
    return False


def main():
    # 启用梯度异常检测，当出现nan/inf梯度时提供详细堆栈跟踪
    torch.autograd.set_detect_anomaly(True)
    if not torch.cuda.is_available():
        logging.info('无可用GPU设备，退出。')
        sys.exit(1)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(f"使用设备: {device}")

    # 初始化混合精度训练
    scaler_gen = torch.amp.GradScaler('cuda', enabled=False, growth_interval=200)
    scaler_disc = torch.amp.GradScaler('cuda', enabled=False, growth_interval=200)

    # 设置随机种子
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('使用GPU设备 = %s' % args.gpu)
    logging.info("参数 = %s", args)

    # 初始化模型
    model = Network()
    model.enhance.init_conv.apply(model.enhance_weights_init)
    for block in model.enhance.blocks:
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                layer.apply(model.enhance_weights_init)
    model.enhance.final_conv.apply(model.enhance_weights_init)
    model = model.to(device)
    torch.set_default_dtype(torch.float32)  # 确保默认数据类型为float32
    # 添加模型参数初始化检查与修正
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            logging.warning(f"参数 {name} 包含NaN或Inf值，重新初始化")
            # 使用xavier均匀分布重新初始化有问题的参数
            nn.init.xavier_uniform_(param.data)
    print_memory_usage("模型初始化后（含参数）")

    # 初始化EMA
    ema = EMA(model, decay=0.999)
    ema.register()

    # 初始化指标模型
    lpips_model = lpips_lib.LPIPS(net='alex').to(device)
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    model._criterion = model._criterion.to(device)

    # 初始化早停机制
    early_stopping = EarlyStopping(patience=1000, min_delta=0.01, warmup_epochs=500)

    # 优化器 - 修正参数绑定问题
    generator_params = []
    for name, param in model.named_parameters():
        if not name.startswith('discriminator'):  # 排除判别器参数
            generator_params.append(param)

        # 为判别器设置更高的学习率 (TTUR)
        generator_optimizer = AdamP(generator_params, lr=args.lr_gen, betas=(0.9, 0.999), weight_decay=1e-4)
        discriminator_optimizer = AdamP(model.discriminator.parameters(), lr=args.lr_disc, betas=(0.5, 0.999),
                                        weight_decay=1e-4)

        # +++ 修改：学习率调度器 - 调整T_max以加速收敛 +++
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            generator_optimizer,
            T_max=1000,  # 从2000减少到1000，加速收敛
            eta_min=1e-7
        )
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
            discriminator_optimizer,
            T_max=500,  # 从1000减少到500
            eta_min=1e-6
        )

    # 加载数据集
    train_low_dir = './data/LOL-V1/lol_dataset/eval15/cs/low'
    train_target_dir = './data/LOL-V1/lol_dataset/eval15/cs/high'
    test_low_dir = './data/LOL-V1/lol_dataset/eval15/cs/low'
    test_target_dir = './data/LOL-V1/lol_dataset/eval15/cs/high'

    TestDataset = DataLoader(img_dir=test_low_dir, target_dir=test_target_dir, task='test')
    TrainDataset = DataLoader(img_dir=train_low_dir, target_dir=train_target_dir, task='train')

    # 打印模型参数量
    MB = utils.count_parameters_in_MB(model)
    logging.info("模型参数量 = %f MB", MB)
    print(f"Model Parameters: {MB:.2f} MB")

    # 创建数据加载器
    train_queue = TorchDataLoader(TrainDataset, batch_size=args.batch_size, pin_memory=False, num_workers=0,
                                  shuffle=True)
    test_queue = TorchDataLoader(TestDataset, batch_size=1, pin_memory=False, num_workers=0, shuffle=False)

    # 初始化指标日志文件
    metrics_log_path = os.path.join(args.save, 'training_metrics.csv')
    # 确保目录存在
    metrics_dir = os.path.dirname(metrics_log_path)
    os.makedirs(metrics_dir, exist_ok=True)
    # 初始化CSV文件并写入表头（仅当文件不存在时）
    if not os.path.exists(metrics_log_path):
        with open(metrics_log_path, 'w') as f:
            f.write("Epoch,Image_Name,PSNR,SSIM,LPIPS,NIQE\n")

    # 初始化详细指标日志文件
    detailed_metrics_path = os.path.join(args.save, 'detailed_metrics.csv')
    if not os.path.exists(detailed_metrics_path):
        with open(detailed_metrics_path, 'w') as f:
            f.write("Epoch,PSNR,SSIM,LPIPS,NIQE\n")

    total_step = 0
    model.train()
    best_composite_score = -float('inf')

    # 添加梯度监控函数
    def get_grad_norms(model, layer_names):
        """获取指定层的梯度范数"""
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None and any(layer_name in name for layer_name in layer_names):
                grad_norms[name] = param.grad.data.norm(2).item()
        return grad_norms

    # 指定要监控的层
    monitor_layers = ['enhance', 'denoise_1', 'denoise_2', 'discriminator']

    try:
        for epoch in range(args.epochs):
            if epoch < 100:
                disc_update_freq = 5  # 训练初期：每5步更新1次（减少判别器压制）
            elif epoch < 500:
                disc_update_freq = 3  # 训练中期：每3步更新1次（平衡对抗）
            else:
                disc_update_freq = 2  # 训练后期：每2步更新1次（正常对抗）
                # 新增2：记录当前更新频率，方便调试
            logging.info(f"Epoch {epoch} | 判别器更新频率: 每{disc_update_freq}步更新1次")

            # 应用课程学习策略
            loss_weights = adjust_loss_weights(epoch)
            # 新增：获取当前噪声水平


            # 更新损失函数中的权重 - 使用新的权重字典
            model._criterion.current_weights = {
                'pixel_reconstruction': loss_weights['pixel_reconstruction'],
                'perceptual': loss_weights['perceptual'],
                'texture_preserve': loss_weights['texture_preserve'],
                'color_constancy': loss_weights['color_constancy'],
                'histogram_match': loss_weights['histogram_match'],
                'ms_ssim': loss_weights['ms_ssim'],
                'frequency': loss_weights['frequency'],
                'noise_aware': loss_weights['noise_aware'],
                'psnr': loss_weights['psnr']
            }
            model._criterion.overexposure_weight = loss_weights['overexposure_weight']
            args.adv_weight = loss_weights['adv_weight']

            losses_gen = []
            losses_disc = [0.0]

            for idx, (input, target, img_name) in enumerate(train_queue):
                total_step += 1
                input = input.to(device).requires_grad_(True)
                target = target.to(device)
                # 新增：根据当前噪声水平添加高斯噪声到输入
                # +++ 修改：应用更真实的退化作为课程学习 +++
                # 随着epoch增加，退化程度从0线性增加到1（前2000个epoch达到最大）
                degradation_severity = min(1.0, epoch / 2000.0)
                logging.info(f"Epoch {epoch} 退化严重程度: {degradation_severity:.4f}")
                input_degraded = utils.degrade_image(input, degradation_severity)

                # 1. 训练判别器 - 每disc_update_freq步训练一次
                # 修改判别器训练部分
                if (total_step % disc_update_freq == 0) and (len(losses_disc) > 0 and abs(losses_disc[-1]) < 100):
                    try:
                        discriminator_optimizer.zero_grad()

                        with torch.amp.autocast('cuda'):
                            with torch.no_grad():
                                outputs = model(input_degraded)
                            pred_img = outputs['H2'].detach()  # 从生成器分离，避免梯度传回

                            # 准备判别器的输入，确保范围正确且仅变换一次
                            # 真实图像从  变换到 [-1, 1]
                            real_input = torch.clamp(target * 2 - 1, -1.0, 1.0)
                            fake_input = torch.clamp(pred_img * 2 - 1, -1.0, 1.0)

                            # 为安全起见，进行钳位操作
                            real_input = torch.clamp(real_input, -1.0, 1.0)
                            fake_input = torch.clamp(fake_input, -1.0, 1.0)

                            # 判别器前向传播
                            real_pred = model.discriminator(real_input)
                            fake_pred = model.discriminator(fake_input)

                            # 添加数值稳定性处理
                            real_pred = torch.clamp(real_pred, -10, 10)
                            fake_pred = torch.clamp(fake_pred, -10, 10)

                            # 记录判别器预测分布
                            disc_stats = {
                                'real_mean': real_pred.mean().item(),
                                'real_std': real_pred.std().item(),
                                'real_range': [real_pred.min().item(), real_pred.max().item()],
                                'fake_mean': fake_pred.mean().item(),
                                'fake_std': fake_pred.std().item(),
                                'fake_range': [fake_pred.min().item(), fake_pred.max().item()]
                            }

                            # 记录判别器梯度
                            disc_grad_norms = get_grad_norms(model.discriminator, monitor_layers)

                            # 记录数据信息
                            data_info = {
                                'batch_index': idx,
                                'total_batches': len(train_queue),
                                'input_range': [input.min().item(), input.max().item()],
                                'target_range': [target.min().item(), target.max().item()],
                                'input_degraded_range': [input_degraded.min().item(), input_degraded.max().item()],

                            }

                            # 计算WGAN-GP损失 - 添加数值稳定性处理,绝对值
                            disc_loss = torch.mean(real_pred) - torch.mean(fake_pred)
                            # 添加梯度惩罚
                            # +++ 修改：应用0-GP到真实样本 +++
                            gradient_penalty = compute_gradient_penalty(model.discriminator, real_input)
                            disc_loss = disc_loss + args.gradient_penalty_weight * gradient_penalty  # 使用新的权重参数

                            # 添加判别器损失正则化
                            disc_regularization = 0.001 * torch.mean(real_pred ** 2)
                            disc_loss = disc_loss + disc_regularization

                            # 检查损失有效性
                            if torch.isnan(disc_loss) or torch.isinf(disc_loss):
                                logging.warning("Invalid discriminator loss, skipping update")
                                discriminator_optimizer.zero_grad()
                                continue

                            # 反向传播和优化
                            scaler_disc.scale(disc_loss).backward()
                            scaler_disc.unscale_(discriminator_optimizer)
                            # 添加梯度裁剪
                            torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_value_(model.discriminator.parameters(), clip_value=0.5)
                            scaler_disc.step(discriminator_optimizer)
                            scaler_disc.update()
                            # 更严格的梯度裁剪




                            losses_disc.append(disc_loss.item())

                            # 每10步记录详细信息
                            if total_step % 10 == 0:
                                logging.info(f"Discriminator Stats: {disc_stats}")
                                logging.info(f"Discriminator Grad Norms: {disc_grad_norms}")
                                logging.info(f"Data Info: {data_info}")

                            logging.info(f"Discriminator trained successfully, loss: {disc_loss.item():.6f}")
                            if gen_loss_val > 1000:
                                logging.warning(
                                    f"Anomaly detected at step {total_step}: Gen loss = {gen_loss_val}, Image = {img_name}")
                                # Also log the breakdown during anomalies
                                detailed_losses = model._criterion.get_detailed_loss_components()
                                logging.warning(f"Anomaly Loss Breakdown: {detailed_losses}")

                    except Exception as e:
                        logging.error(f"Error training discriminator: {e}")
                        # 重置梯度，防止累积
                        discriminator_optimizer.zero_grad()
                        # 跳过本次更新但记录一个合理值
                        losses_disc.append(1.0)  # 使用中性值而不是0
                        continue
                else:
                    # 即使不更新判别器，也记录上一次损失（避免列表为空）
                    if losses_disc:  # 列表非空时记录上一次值
                        losses_disc.append(losses_disc[-1])
                    else:
                        losses_disc.append(0.0)  # 初始值
                # 2. 训练生成器
                generator_optimizer.zero_grad()

                # 初始化变量，避免未定义错误
                pred = None
                fake_pred = None
                content_loss = None
                adv_loss = None
                gen_loss = None
                outputs = None

                # 仅调用一次model(input)，复用输出
                with torch.amp.autocast('cuda'):
                    outputs = model(input_degraded)
                    for key, tensor in outputs.items():
                        if torch.is_tensor(tensor) and (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
                            logging.error(f"NaN/Inf found in {key} at step {total_step}. Skipping batch.")
                            continue  # 跳过这个batch


                # 直接使用修复后的outputs，不再重复调用
                    pred = outputs['H2']
                    pred = torch.clamp(pred, 0, 1)  # 确保在[0,1]范围
                    # 转换到[-1,1]范围再输入判别器
                    pred_disc = pred * 2 - 1
                    fake_pred_g = model.discriminator(pred_disc)

                    logging.info(
                    f"[DEBUG] H2 - min: {pred.min().item():.6f}, max: {pred.max().item():.6f}, mean: {pred.mean().item():.6f}")

                    gen_content_loss = model._loss(input, target, epoch=epoch, **outputs)  # 内部已按current_weights加权
                    if torch.isnan(gen_content_loss) or torch.isinf(gen_content_loss):
                        logging.warning("Invalid content loss, skipping batch")
                        continue
                      # 使用新的权重键
                    gen_content_loss = gen_content_loss * loss_weights['pixel_reconstruction']

                    pred_disc = torch.clamp(pred * 2 - 1, -1.0, 1.0)
                    fake_pred_g = model.discriminator(pred_disc)

                    # 添加数值稳定性处理
                    fake_pred_g = torch.clamp(fake_pred_g, -10, 10)
                    if torch.isnan(fake_pred_g).any() or torch.isinf(fake_pred_g).any():
                        logging.warning("NaN or Inf in fake_pred_g")
                        continue
                    gen_adv_loss = -torch.mean(fake_pred_g)

                    # 总生成器损失（使用动态调整的对抗权重）
                    gen_loss = gen_content_loss + args.adv_weight * gen_adv_loss

                    # 检查生成器损失是否有NaN或Inf
                    if check_nan_inf(gen_loss, "gen_loss"):
                        generator_optimizer.zero_grad()
                        continue # 跳过这个batch

                # 确保gen_loss是张量
                    if not torch.is_tensor(gen_loss):
                        gen_loss = torch.tensor(gen_loss, device=device, dtype=torch.float32, requires_grad=True)

                # 获取详细损失分量（如果可用）
                try:
                    detailed_loss = model._criterion.get_detailed_loss_components()
                except:
                    detailed_loss = "Not available"

                # 获取中间层输出（如果可用）
                try:
                    intermediate_outputs = model.get_intermediate_outputs()
                except:
                    intermediate_outputs = "Not available"

                # 获取生成器梯度
                gen_grad_norms = get_grad_norms(model, monitor_layers)

                # 记录学习率
                lr_info = {
                    'gen_expected': scheduler_gen.get_last_lr()[0],
                    'gen_actual': generator_optimizer.param_groups[0]['lr'],
                    'disc_expected': scheduler_disc.get_last_lr()[0],
                    'disc_actual': discriminator_optimizer.param_groups[0]['lr']
                }

                # 每10步记录详细信息
                if total_step % 10 == 0:
                    logging.info(f"Generator Loss Breakdown: {detailed_loss}")
                    logging.info(f"Intermediate Outputs: {intermediate_outputs}")
                    logging.info(f"Generator Grad Norms: {gen_grad_norms}")
                    logging.info(f"Learning Rate Info: {lr_info}")

                    # 记录参数更新量
                    param_update_norms = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None and any(layer_name in name for layer_name in monitor_layers):
                            update_norm = (param.grad.data * generator_optimizer.param_groups[0]['lr']).norm(
                                2).item()
                            param_update_norms[name] = update_norm
                    logging.info(f"Parameter Update Norms: {param_update_norms}")

                # 保存损失值用于日志
                gen_loss_val = gen_loss.item()
                # +++ 新增：在反向传播前检查gen_loss的有效性 +++
                if check_nan_inf(gen_loss, "gen_loss"):
                    generator_optimizer.zero_grad()
                    continue  # 跳过这个batch
                # 反向传播和优化生成器
                scaler_gen.scale(gen_loss).backward()
                scaler_gen.unscale_(generator_optimizer)

                # 使用更温和的梯度裁剪
                torch.nn.utils.clip_grad_norm_(generator_params, max_norm=0.8)  # 收紧最大范数


                # 设置更保守的自动混合精度

                scaler_gen.step(generator_optimizer)  # 优化器步骤
                scaler_gen.update()  # 混合精度更新

                losses_gen.append(gen_loss_val)

                # 每10步打印详细日志
                if total_step % 10 == 0:
                    # 记录判别器损失（如果已计算）
                    disc_loss_val = losses_disc[-1] if losses_disc else 0.0
                    logging.info('epoch %d step %d gen_loss %f disc_loss %f',
                                 epoch, total_step, gen_loss_val, disc_loss_val)

                    # 记录各损失组件
                    if hasattr(model._criterion, 'current_weights'):
                        logging.info(f"损失权重: {model._criterion.current_weights}")

                    # 记录亮度统计
                    if hasattr(model._criterion, 'avg_brightness'):
                        logging.info(
                            f"平均亮度: {model._criterion.avg_brightness:.4f}, 过曝比例: {model._criterion.overexposure_ratio:.4f}")

                    # 记录学习率
                    current_lr_gen = generator_optimizer.param_groups[0]['lr']
                    current_lr_disc = discriminator_optimizer.param_groups[0]['lr']
                    logging.info(f"学习率 - 生成器: {current_lr_gen:.2e}, 判别器: {current_lr_disc:.2e}")

                    # 记录噪声分类结果
                    if 'noise_prob' in outputs:
                        noise_prob = outputs['noise_prob']
                        logging.info(
                            f"噪声概率 - 高斯: {noise_prob[0, 0]:.3f}, 泊松: {noise_prob[0, 1]:.3f}, 椒盐: {noise_prob[0, 2]:.3f}")

                # 更新EMA
                ema.update()

                # 清理显存
                if total_step % 2 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # 每50步监控梯度
                if total_step % 50 == 0:
                    # 监控梯度范数
                    total_grad_norm = 0
                    grad_norms = []
                    for name, param in model.named_parameters():
                        if param.grad is not None and "generator" in name:
                            param_grad_norm = param.grad.data.norm(2).item()
                            grad_norms.append((name, param_grad_norm))
                            total_grad_norm += param_grad_norm ** 2

                            # 修复过小的梯度（梯度消失）
                            if param_grad_norm < 1e-8:
                                logging.warning(f"梯度消失检测: {name}, 范数: {param_grad_norm:.8f}")
                                # 添加少量噪声重启梯度
                                param.grad.data += torch.randn_like(param.grad.data) * 1e-6

                            # 修复过大的梯度（梯度爆炸）
                            if param_grad_norm > 1000:
                                logging.warning(f"梯度爆炸检测: {name}, 范数: {param_grad_norm:.2f}")
                                torch.nn.utils.clip_grad_norm_([param], max_norm=10.0)

                    total_grad_norm = total_grad_norm ** 0.5
                    logging.info(f'总梯度范数: {total_grad_norm:.6f}')

                    # 记录前5个最大梯度
                    grad_norms.sort(key=lambda x: x[1], reverse=True)
                    for i, (name, norm) in enumerate(grad_norms[:5]):
                        logging.info(f'梯度TOP{i + 1}: {name} = {norm:.6f}')

                    # 监控参数更新量
                    param_update_norm = 0
                    for p in generator_params:
                        if p.grad is not None:
                            param_update_norm += (p.grad.data * generator_optimizer.param_groups[0]['lr']).norm(
                                2).item() ** 2
                    param_update_norm = param_update_norm ** 0.5

                    logging.info(f'参数更新量: {param_update_norm:.8f}')
                    try:
                        # 判别器对真实样本的输出范围
                        if 'real_pred' in locals():
                            logging.info(f"Real pred range: [{real_pred.min():.3f}, {real_pred.max():.3f}]")
                        # 判别器对生成样本的输出范围
                        if 'fake_pred' in locals():
                            logging.info(f"Fake pred range: [{fake_pred.min():.3f}, {fake_pred.max():.3f}]")
                        # 生成器输出的数值范围
                        if 'pred' in locals():
                            logging.info(f"Gen output range: [{pred.min():.3f}, {pred.max():.3f}]")
                    except Exception as e:
                        logging.warning(f"监控输出范围时出错: {e}")

                # 每50步记录batch级指标
                if total_step % 50 == 0 and target is not None:
                    with torch.no_grad():
                        psnr_val = psnr_metric(outputs['H2'], target)
                        ssim_val = ssim_metric(outputs['H2'], target)
                        logging.info(f"Batch {idx} Metrics - PSNR: {psnr_val.item():.4f}, SSIM: {ssim_val.item():.4f}")

                # 异常检测
                if gen_loss_val > 1000:  # 异常阈值
                    logging.warning(
                        f"Anomaly detected at step {total_step}: Gen loss = {gen_loss_val}, Image = {img_name}")

                # 每100步检查参数和梯度的数值稳定性（NaN/Inf/范围）
                if total_step % 100 == 0:
                    # 检查参数NaN/Inf
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            logging.warning(f"NaN detected in parameter: {name}")
                        if torch.isinf(param).any():
                            logging.warning(f"Inf detected in parameter: {name}")

                    # 检查梯度NaN/Inf
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                logging.warning(f"NaN detected in gradient: {name}")
                            if torch.isinf(param.grad).any():
                                logging.warning(f"Inf detected in gradient: {name}")

                    # 检查参数范围（避免数值爆炸）
                    for name, param in model.named_parameters():
                        if param.numel() > 0:  # 跳过空参数
                            param_min = param.min().item()
                            param_max = param.max().item()
                            if abs(param_max) > 1e4 or abs(param_min) > 1e4:
                                logging.warning(
                                    f"Parameter {name} has large values: min={param_min:.4f}, max={param_max:.4f}")

                # 清理变量，只删除已定义的变量
                variables_to_delete = ['pred', 'fake_pred', 'content_loss', 'adv_loss', 'gen_loss', 'outputs']
                for var_name in variables_to_delete:
                    if var_name in locals():
                        del locals()[var_name]
                torch.cuda.empty_cache()
                gc.collect()

            # 添加损失列表空值检查
            if not losses_gen:
                logging.warning(f"Epoch {epoch}: 生成器损失列表为空，可能训练步骤被跳过")
                continue

            if not losses_disc:
                logging.warning(f"Epoch {epoch}: 判别器损失列表为空，可能训练步骤被跳过")
                continue

            # 更新学习率
            mean_gen_loss = np.mean(losses_gen) if losses_gen else 0.0
            mean_disc_loss = np.mean(losses_disc) if losses_disc else 0.0

            current_lr_gen = generator_optimizer.param_groups[0]['lr']
            current_lr_disc = discriminator_optimizer.param_groups[0]['lr']

            logging.info(f"Current Learning Rates - Gen: {current_lr_gen}, Disc: {current_lr_disc}")

            # 使用生成器损失作为监控指标
            scheduler_gen.step()  # 余弦退火调度器在每个epoch结束时更新，无需传入参数
            scheduler_disc.step()

            # 验证和保存
            if total_step != 0 and epoch % 2 == 0:  # 每2个epoch验证一次以节省时间
                ema.apply_shadow()
                model.eval()
                epoch_metrics = []
                composite_scores = []

                # 记录更详细的验证指标
                epoch_metrics_detailed = []

                with torch.no_grad():
                    for idx, (input, target, img_name) in enumerate(test_queue):
                        input = Variable(input).to(device)
                        target = Variable(target).to(device)

                        image_name = os.path.splitext(os.path.basename(img_name[0]))[0]

                        outputs = model(input)
                        enhanced_H3 = outputs['H3']
                        enhanced_H2 = outputs['H2']

                        # 应用自适应亮度控制并确保范围正确
                        enhanced_H2 = adaptive_brightness_control(enhanced_H2)
                        enhanced_H3 = torch.clamp(enhanced_H3, 0, 1)
                        enhanced_H2 = torch.clamp(enhanced_H2, 0, 1)

                        # 确保目标图像也在正确范围内
                        target = torch.clamp(target, 0, 1)

                        # 计算指标 - 确保输入范围正确
                        # PSNR和SSIM需要确保输入在[0,1]范围内
                        psnr_value = psnr_metric(enhanced_H2, target)
                        ssim_value = ssim_metric(enhanced_H2, target)

                        # LPIPS需要将输入从[0,1]转换到[-1,1]
                        lpips_input = enhanced_H2 * 2 - 1  # [0,1] -> [-1,1]
                        lpips_target = target * 2 - 1  # [0,1] -> [-1,1]
                        lpips_value = lpips_model(lpips_input, lpips_target).mean()

                        # NIQE只需要增强后的图像
                        niqe_value = niqe_metric(enhanced_H2)

                        epoch_metrics.append({
                            'name': image_name,
                            'psnr': psnr_value.item(),
                            'ssim': ssim_value.item(),
                            'lpips': lpips_value.item(),
                            'niqe': niqe_value.item()
                        })

                        metrics = {
                            'name': image_name,
                            'psnr': psnr_value.item(),
                            'ssim': ssim_value.item(),
                            'lpips': lpips_value.item(),
                            'niqe': niqe_value.item(),
                            # 记录验证时的中间输出（如果可用）
                            'intermediate_outputs': model.get_intermediate_outputs() if hasattr(model,
                                                                                                'get_intermediate_outputs') else "Not available"
                        }
                        epoch_metrics_detailed.append(metrics)

                        # 计算综合得分
                        composite_score = (psnr_value.item() / 40 * 0.4 +  # PSNR归一化
                                           ssim_value.item() * 0.4 +  # SSIM
                                           (1 - lpips_value.item()) * 0.1 +  # LPIPS反向
                                           (1 - min(niqe_value.item() / 10, 1)) * 0.1)  # NIQE归一化
                        composite_scores.append(composite_score)

                        with open(metrics_log_path, 'a') as f:
                            f.write(
                                f"{epoch},{image_name},{psnr_value.item():.4f},{ssim_value.item():.4f},{lpips_value.item():.4f},{niqe_value.item():.4f}\n")
                            f.flush()  # 确保数据立即写入磁盘，避免缓存导致的数据丢失

                        # 在验证循环中添加以下调试代码
                        if epoch % 2 == 0:
                            # 添加输入和目标图像的统计信息
                            logging.info(f"输入图像范围: [{input.min().item():.4f}, {input.max().item():.4f}]")
                            logging.info(f"目标图像范围: [{target.min().item():.4f}, {target.max().item():.4f}]")
                            logging.info(
                                f"增强图像范围: [{enhanced_H2.min().item():.4f}, {enhanced_H2.max().item():.4f}]")

                            # 检查PSNR计算是否正确
                            mse = F.mse_loss(enhanced_H2, target)
                            manual_psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
                            if abs(psnr_value.item() - manual_psnr.item()) > 0.1:
                                logging.warning(
                                    f"PSNR计算不一致: 库计算={psnr_value.item():.4f}, 手动计算={manual_psnr.item():.4f}")
                        # 定期保存图像
                        if epoch % 50 == 0:
                            H3_img = save_images(enhanced_H3)
                            denoise_dir = os.path.join(args.save, 'result/denoise')
                            os.makedirs(denoise_dir, exist_ok=True)
                            Image.fromarray(H3_img).save(os.path.join(denoise_dir, f"{image_name}_denoise_{epoch}.png"),
                                                         'PNG')

                            H2_img = save_images(enhanced_H2)
                            enhance_dir = os.path.join(args.save, 'result/enhance')
                            os.makedirs(enhance_dir, exist_ok=True)
                            Image.fromarray(H2_img).save(os.path.join(enhance_dir, f"{image_name}_enhance_{epoch}.png"),
                                                         'PNG')

                # 记录平均指标
                avg_metrics = {
                    'psnr': np.mean([m['psnr'] for m in epoch_metrics_detailed]),
                    'ssim': np.mean([m['ssim'] for m in epoch_metrics_detailed]),
                    'lpips': np.mean([m['lpips'] for m in epoch_metrics_detailed]),
                    'niqe': np.mean([m['niqe'] for m in epoch_metrics_detailed])
                }

                logging.info(f"Epoch {epoch} Detailed Metrics: {avg_metrics}")

                # 保存详细指标到文件
                with open(detailed_metrics_path, 'a') as f:
                    f.write(
                        f"{epoch},{avg_metrics['psnr']},{avg_metrics['ssim']},{avg_metrics['lpips']},{avg_metrics['niqe']}\n")

                ema.restore()  # 恢复原始权重
                model.train()  # 切换回训练模式
                # 计算平均指标
                avg_psnr = np.mean([m['psnr'] for m in epoch_metrics]) if epoch_metrics else 0
                avg_ssim = np.mean([m['ssim'] for m in epoch_metrics]) if epoch_metrics else 0
                avg_lpips = np.mean([m['lpips'] for m in epoch_metrics]) if epoch_metrics else 0
                avg_niqe = np.mean([m['niqe'] for m in epoch_metrics]) if epoch_metrics else 0
                avg_composite = np.mean(composite_scores) if composite_scores else 0

                logging.info(
                    f"Epoch {epoch} Metrics - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}, NIQE: {avg_niqe:.4f}, Composite: {avg_composite:.4f}")

                # 检查早停 - 添加current_epoch参数
                if early_stopping(avg_composite, epoch):
                    logging.info(f"早停触发于 epoch {epoch}")
                    break

                # 保存模型（当PSNR指标大于20时保存）
                if avg_psnr > 20:
                    torch.save(model.state_dict(), os.path.join(model_path, f'best_model.pt'))
                    logging.info(f"PSNR大于20，保存模型，当前PSNR: {avg_psnr:.4f}")

                ema.restore()
                model.train()

                gc.collect()
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated() / 1024 ** 3  # 已分配显存（GB）
                reserved = torch.cuda.memory_reserved() / 1024 ** 3  # 已预留显存（GB）
                logging.info('epoch %d GPU Memory - Allocated: %.2fGB, Reserved: %.2fGB',
                             epoch, alloc, reserved)

            # 修改检查点保存条件
            avg_psnr = np.mean([m['psnr'] for m in epoch_metrics])
            if epoch % 100 == 0 and avg_psnr > 18:  # 只在PSNR>18时保存
                # 保存完整训练状态（包含模型、优化器、调度器等）
                checkpoint = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'gen_optimizer': generator_optimizer.state_dict(),
                    'disc_optimizer': discriminator_optimizer.state_dict(),
                    'gen_scaler': scaler_gen.state_dict(),  # 混合精度缩放器状态
                    'disc_scaler': scaler_disc.state_dict(),
                    'gen_scheduler': scheduler_gen.state_dict(),  # 学习率调度器状态
                    'disc_scheduler': scheduler_disc.state_dict(),
                    'gen_losses': losses_gen,  # 当前epoch生成器损失
                    'disc_losses': losses_disc,  # 当前epoch判别器损失
                    'ema_state': ema.shadow,  # EMA模型状态
                    'best_composite_score': best_composite_score  # 最佳综合得分
                }
                # 确保保存路径存在
                checkpoint_path = os.path.join(model_path, f'checkpoint_epoch_{epoch}.pt')
                torch.save(checkpoint, checkpoint_path)
                logging.info(f"Saved full checkpoint to {checkpoint_path}")
            # 内存监控
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"Epoch {epoch} | 已分配: {alloc:.2f}GB | 已预留: {reserved:.2f}GB")

    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
    finally:
        # 保存最终模型
        torch.save(model.state_dict(), os.path.join(model_path, 'final_model.pt'))

        # 分析指标并写入日志
        log_path = os.path.join(args.save, 'log.txt')
        write_best_metrics_to_log(metrics_log_path, log_path)
        logging.info("训练结束，最佳指标已写入日志文件首行")


if __name__ == '__main__':
    main()