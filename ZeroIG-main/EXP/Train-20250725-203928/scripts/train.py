import os

# 调整CUDA内存分配策略，减少内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

import sys
import time
import glob
import numpy as np
import utils
from PIL import Image
import logging
import argparse
import torch
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
# 导入PyTorch自动混合精度模块
from torch.cuda.amp import GradScaler, autocast
from model import *
from multi_read_data import DataLoader
# 导入梯度检查点工具（如需使用）
from torch.utils.checkpoint import checkpoint

parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=800, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--save', type=str, default='./EXP/', help='location of the data corpus')
parser.add_argument('--model_pretrain', type=str, default='', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor):
    """将张量转换为可保存的图像格式"""
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)

    # 初始化模型
    model = Network()
    utils.save(model, os.path.join(args.save, 'initial_weights.pt'))
    # 初始化模型权重
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)
    model = model.cuda()

    # 初始化优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=3e-4
    )

    # 初始化混合精度训练所需的梯度缩放器
    scaler = GradScaler()

    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)

    # 加载数据集
    train_low_data_names = './data/indoor/dark-1'
    TrainDataset = DataLoader(img_dir=train_low_data_names, task='train')

    test_low_data_names = './data/indoor/dark-2'
    TestDataset = DataLoader(img_dir=test_low_data_names, task='test')

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=0, shuffle=False, generator=torch.Generator(device='cuda'))

    total_step = 0
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for idx, (input, img_name) in enumerate(train_queue):
            total_step += 1
            input = Variable(input, requires_grad=False).cuda()
            optimizer.zero_grad()

            # 设置优化器可捕获性，用于AMP
            optimizer.param_groups[0]['capturable'] = True

            # 自动混合精度上下文管理器（确保正确启用）
            with autocast():
                # 前向传播计算损失（使用混合精度）
                loss = model._loss(input)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), 5)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 清理未使用的张量和CUDA缓存
            losses.append(loss.item())
            del input, loss  # 删除当前迭代不再需要的变量
            torch.cuda.empty_cache()  # 强制清理CUDA缓存

            logging.info('train-epoch %03d %03d %f', epoch, idx, losses[-1])

        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        utils.save(model, os.path.join(model_path, 'weights_%d.pt' % epoch))

        # 每50个epoch进行一次测试
        if epoch % 50 == 0 and total_step != 0:
            model.eval()
            with torch.no_grad():
                with autocast():  # 测试时也启用混合精度
                    for idx, (input, img_name) in enumerate(test_queue):
                        input = Variable(input, volatile=True).cuda()
                        image_name = img_name[0].split('/')[-1].split('.')[0]
                        # 模型推理（如需进一步节省内存，可对模型forward使用checkpoint）
                        outputs = model(input)  # 简化为单变量接收所有输出
                        L_pred1, L_pred2, L2, s2, s21, s22, H2, H11, H12, H13, s13, H14, s14, H3, s3, H3_pred, H4_pred, L_pred1_L_pred2_diff, H13_H14_diff, H2_blur, H3_blur = outputs

                        # 保存图像
                        input_name = '%s' % (image_name)
                        H3_img = save_images(H3)
                        H2_img = save_images(H2)
                        os.makedirs(args.save + '/result/denoise/', exist_ok=True)
                        os.makedirs(args.save + '/result/enhance/', exist_ok=True)
                        Image.fromarray(H3_img).save(
                            args.save + '/result/denoise/' + input_name + '_denoise_' + str(epoch) + '.png', 'PNG')
                        Image.fromarray(H2_img).save(
                            args.save + '/result/enhance/' + input_name + '_enhance_' + str(epoch) + '.png', 'PNG')

                        # 清理测试过程中的中间变量
                        del input, outputs, H3, H2, H3_img, H2_img
                        torch.cuda.empty_cache()
            model.train()


if __name__ == '__main__':
    main()