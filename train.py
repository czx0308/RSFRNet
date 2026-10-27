# -----------------------------------------------------------
# File: train.py (Fixed Logic + Original Visualization)
# -----------------------------------------------------------
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import warnings
import numpy as np

# 忽略所有警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from ptflops import get_model_complexity_info

# --- 导入项目自定义模块 ---
import models
from models.loss2 import CP_ClDiceLoss
from models.loss2A import clDiceLoss, OHEM_BCE_Loss
from referece.PCGrad import PCGrad  # 确保路径名对齐
from utils import *
from edge_dataloader import MultiTaskLoader

# ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Zhaoxiang Cao Gabor')

# --- 参数设置 (保持完整) ---
parser.add_argument('--cv', type=int, default=0, help='cross-validation split id')
parser.add_argument('--class_uniform_pct', type=float, default=0.0)
parser.add_argument('--class_uniform_tile', type=int, default=1024)
parser.add_argument('--img_wt_loss', action='store_true', default=False)
parser.add_argument('--batch_weighting', action='store_true', default=False)
parser.add_argument('--dice_loss', default=False, action='store_true')
parser.add_argument("--ohem", action="store_true", default=False)
parser.add_argument("--aux", action="store_true", default=False)
parser.add_argument('--jointwtborder', action='store_true', default=False)
parser.add_argument('--joint_edge_loss_pfnet', action='store_true')
parser.add_argument('--edge_weight', type=float, default=1.0)
parser.add_argument('--seg_weight', type=float, default=1.0)
parser.add_argument('--strict_bdr_cls', type=str, default='')
parser.add_argument('--rlx_off_epoch', type=int, default=-1)
parser.add_argument('--rescale', type=float, default=1.0)
parser.add_argument('--repoly', type=float, default=1.5)
parser.add_argument('--apex', action='store_true', default=False)
parser.add_argument('--fp16', action='store_true', default=False)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)
parser.add_argument('--freeze_trunk', action='store_true', default=False)
parser.add_argument('--hardnm', default=0, type=int)
parser.add_argument('--trunk', type=str, default='resnet101')
parser.add_argument('--max_epoch', type=int, default=180)
parser.add_argument('--max_cu_epoch', type=int, default=100000)
parser.add_argument('--color_aug', type=float, default=0.0)
parser.add_argument('--gblur', action='store_true', default=True)
parser.add_argument('--bblur', action='store_true', default=False)
parser.add_argument('--lr_schedule', type=str, default='poly')
parser.add_argument('--poly_exp', type=float, default=1.0)
parser.add_argument('--bs_mult', type=int, default=4)
parser.add_argument('--bs_mult_val', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=720)
parser.add_argument('--pre_size', type=int, default=None)
parser.add_argument('--scale_min', type=float, default=1.0)
parser.add_argument('--scale_max', type=float, default=1.0)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default')
parser.add_argument('--tb_tag', type=str, default='')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=False)
parser.add_argument('--fix_bn', action='store_true', default=False)
parser.add_argument('--dump_augmentation_images', action='store_true', default=False)
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
parser.add_argument('--scf', action='store_true', default=False)
parser.add_argument('--print_freq', type=int, default=30)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--with_aug', action='store_true')
parser.add_argument('--thicky', default=8, type=int)
parser.add_argument('--match_dim', default=64, type=int)
parser.add_argument('--ignore_background', action='store_true')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--savedir', type=str, default='results/savedir')
parser.add_argument('--datadir', type=str, default='../data')
parser.add_argument('--only-bsds', action='store_true')
parser.add_argument('--dataset', type=str, default='BSDS')
parser.add_argument('--model', type=str, default='baseline')
parser.add_argument('--sa', action='store_true')
parser.add_argument('--dil', action='store_true')
parser.add_argument('--config', type=str, default='carv4')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--checkinfo', action='store_true')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--iter-size', type=int, default=24)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr-type', type=str, default='multistep')
parser.add_argument('--lr-steps', type=str, default=None)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--eta', type=float, default=0)
parser.add_argument('--lmbda', type=float, default=1.1)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--save-freq', type=int, default=1)
parser.add_argument('--evaluate', type=int, default=None)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def compute_model_info(model, input_size=(3, 384, 384)):
    model.eval()
    try:
        def input_constructor(input_shape):
            dev = next(model.parameters()).device
            x = torch.randn((1, *input_shape)).to(dev)
            cid = torch.zeros((1,), dtype=torch.long).to(dev)
            return {"x": x, "class_id": cid}

        flops, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False,
            verbose=False, input_constructor=input_constructor
        )
        params = params / 1e6
        flops = flops / 1e9
    except Exception as e:
        print(f"FLOPs 计算失败: {str(e)}")
        params, flops = 0, 0

    avg_time_single, avg_time_batch = 0, 0
    return params, flops, avg_time_single, avg_time_batch


def main(running_file):
    global args

    ### Refine args
    if args.seed is None:
        args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.use_cuda = torch.cuda.is_available()

    ### Create model
    model = getattr(models, args.model)(args)

    if args.use_cuda:
        model = model.cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    input_size = (3, 384, 384)
    params, flops, time_single, time_batch = compute_model_info(model, input_size)
    print(f"Model Properties:")
    print(f"  - Parameters: {params:.2f} M")
    print(f"  - FLOPs: {flops:.2f} G")

    if args.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    ### Load Data
    train_dataset = MultiTaskLoader(split="train", threshold=args.eta)
    val_dataset = MultiTaskLoader(split="val", threshold=args.eta)

    train_loader = DataLoader(
        train_dataset, batch_size=8, num_workers=args.workers, shuffle=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=8, num_workers=args.workers, shuffle=False)

    ### Create log file
    os.makedirs(args.savedir, exist_ok=True)
    log_file = os.path.join(args.savedir, '%s_log.txt' % args.model)
    log_file2 = os.path.join(args.savedir, '%s_log_val.txt' % args.model)

    args.start_epoch = 0

    # --- 初始化优化器并使用 PCGrad 包装 ---
    if args.opt == 'adam':
        base_optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    else:
        base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    optimizer = PCGrad(base_optimizer)
    scheduler = StepLR(optimizer, step_size=45, gamma=0.5)

    if args.resume:
        checkpoint = load_checkpoint(args, running_file)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    ### Train Loop
    train_loss = []
    val_losss = []
    val_road_ious = []
    val_farm_ious = []
    last_val_mIoU = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # 1. Train
        tr_avg_loss = train(
            train_loader, model, optimizer, epoch, running_file, args)

        train_loss.append(tr_avg_loss)

        # Log format simplified for main loop, but detail is in train()
        log = "Epoch %03d/%03d: train-loss %s | lr %s | Time %s\n" % \
              (epoch, args.epochs, tr_avg_loss, optimizer.param_groups[0]['lr'], time.strftime('%Y-%m-%d %H:%M:%S'))

        with open(args.savedir + '/loss_epoch_data.txt', 'w') as f:
            f.write(str(train_loss))

        # 2. Validate
        val_loss, val_precision, val_recall, val_f1, road_iou, farm_iou, mean_iou = validate(val_loader, model, args)

        # Log Validation & Checkpoint
        with open(log_file, 'a') as f:
            f.write(log)
            if mean_iou > last_val_mIoU:
                last_val_mIoU = mean_iou
                save_checkpoint2({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, epoch, args.savedir)

        print(f"Epoch {epoch + 1}/{args.epochs} - Val Loss: {val_loss:.4f}")
        print(f"  > Road IoU: {road_iou:.4f} | Farm IoU: {farm_iou:.4f} | mIoU: {mean_iou:.4f}")
        print(f"  > Pre: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")

        val_losss.append(val_loss)
        val_road_ious.append(road_iou)
        val_farm_ious.append(farm_iou)

        log3 = "Epoch %03d/%03d: val_loss %s | lr %s | Time %s\n" % \
               (epoch, args.epochs, val_loss, optimizer.param_groups[0]['lr'], time.strftime('%Y-%m-%d %H:%M:%S'))
        log4 = 'mIoU:{:.4f}, Road_IoU:{:.4f}, Farm_IoU:{:.4f}, pre:{:.4f}, recall:{:.4f}, F1:{:.4f} \n'.format(
            mean_iou, road_iou, farm_iou, val_precision, val_recall, val_f1)

        with open(log_file2, 'a') as f:
            f.write(log3)
            f.write(log4)

        # 保存曲线数据
        with open(args.savedir + '/val_loss_list.txt', 'w') as f:
            f.write(str(val_losss))
        with open(args.savedir + '/val_road_iou_list.txt', 'w') as f:
            f.write(str(val_road_ious))
        with open(args.savedir + '/val_farm_iou_list.txt', 'w') as f:
            f.write(str(val_farm_ious))

        scheduler.step()

    return


def train(train_loader, model, optimizer, epoch, running_file, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    running_file.write('\n%s\n' % str(args))
    running_file.flush()

    # --- [恢复 1] 计算格式化宽度 ---
    wD = len(str(len(train_loader) // args.iter_size))
    wE = len(str(args.epochs))

    end = time.time()
    iter_step = 0
    counter = 0
    loss_value = 0
    optimizer.zero_grad()

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
        image, label, class_id = batch

        data_time.update(time.time() - end)

        if args.use_cuda:
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            class_id = class_id.cuda(non_blocking=True)

        # Forward pass
        outputs = model(image, class_id)

        # --- [Logic Fix] 只取 Logits ---
        if isinstance(outputs, tuple):
            pred_logits = outputs[0]
        else:
            pred_logits = outputs

        # 分离任务索引
        road_indices = (class_id == 0).nonzero(as_tuple=True)[0]
        farm_indices = (class_id == 1).nonzero(as_tuple=True)[0]

        objectives = []
        display_loss = 0.0

        # --- Task 1: 道路 ---
        if len(road_indices) > 0:
            p_road = pred_logits[road_indices]
            t_road = label[road_indices]
            loss_base_road = CP_ClDiceLoss(
                pred=p_road, target=t_road, smooth=1, exponent=2,
                class_weight=None, loss_weight=0.5, reduction='mean', beta=1.1
            ).mean()
            loss_topo = clDiceLoss(p_road, t_road, iter_=5)
            total_loss_road = loss_base_road + 0.3 * loss_topo
            objectives.append(total_loss_road)
            display_loss += total_loss_road.item()

        # --- Task 2: 农田 ---
        if len(farm_indices) > 0:
            p_farm = pred_logits[farm_indices]
            t_farm = label[farm_indices]
            loss_base_farm = CP_ClDiceLoss(
                pred=p_farm, target=t_farm, smooth=1, exponent=2,
                class_weight=None, loss_weight=0.5, reduction='mean', beta=1.1
            ).mean()
            loss_region = OHEM_BCE_Loss(p_farm, t_farm, keep_ratio=0.7)
            total_loss_farm = loss_base_farm + 0.3 * loss_region
            objectives.append(total_loss_farm)
            display_loss += total_loss_farm.item()

        # --- PCGrad Backward ---
        counter += 1
        loss_value += display_loss

        if len(objectives) > 0:
            objectives = [obj / args.iter_size for obj in objectives]
            optimizer.pc_backward(objectives)

        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
            iter_step += 1

            losses.update(loss_value / args.iter_size, args.iter_size)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_value = 0

            # --- [恢复 2] 原始的详细打印格式 ---
            # 只有当达到 print_freq 时才打印详细信息到控制台和文件
            if iter_step % args.print_freq == 1:
                runinfo = str(('Epoch: [{0:0%dd}/{1:0%dd}][{2:0%dd}/{3:0%dd}]\t' \
                               % (wE, wE, wD, wD) + \
                               'Time {batch_time.val:.3f}\t' + \
                               'Data {data_time.val:.3f}\t' + \
                               'Loss {loss.val:.4f} (avg:{loss.avg:.4f})\t').format(
                    epoch, args.epochs, iter_step, len(train_loader) // args.iter_size,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, lr=args.lr))

                # 打印到控制台
                print(runinfo)

                # 写入到文件
                running_file.write('%s\n' % runinfo)
                running_file.flush()

    return '%.4f' % (losses.avg)


def accuracy(output, target):
    output = output.copy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    TP = (np.multiply(output, target)).sum()
    FP = (np.multiply(output, (1 - target))).sum()
    FN = (np.multiply((1 - output), target)).sum()
    return TP, FP, FN


def validate(val_loader, model, args):
    batch_time = AverageMeter()
    losses = AverageMeter()

    metrics = {
        0: {'TP': 0, 'FP': 0, 'FN': 0},
        1: {'TP': 0, 'FP': 0, 'FN': 0}
    }

    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image, label, class_id = batch

            if args.use_cuda:
                image = image.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                class_id = class_id.cuda(non_blocking=True)

            outputs = model(image, class_id)

            if isinstance(outputs, tuple):
                pred_logits = outputs[0]
            else:
                pred_logits = outputs

            pred_prob = pred_logits
            predict_numpy = pred_prob.detach().cpu().numpy()

            label_numpy = label.cpu().numpy()
            class_id_numpy = class_id.cpu().numpy()

            # Road Metrics
            road_idx = (class_id_numpy == 0)
            if np.any(road_idx):
                tp, fp, fn = accuracy(predict_numpy[road_idx], label_numpy[road_idx])
                metrics[0]['TP'] += tp
                metrics[0]['FP'] += fp
                metrics[0]['FN'] += fn

            # Farm Metrics
            farm_idx = (class_id_numpy == 1)
            if np.any(farm_idx):
                tp, fp, fn = accuracy(predict_numpy[farm_idx], label_numpy[farm_idx])
                metrics[1]['TP'] += tp
                metrics[1]['FP'] += fp
                metrics[1]['FN'] += fn

            # Loss (using logits is correct for CP_ClDiceLoss)
            loss = CP_ClDiceLoss(pred_logits, label)
            losses.update(loss.mean().item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    def get_iou(tp, fp, fn):
        denom = tp + fp + fn
        return tp / denom if denom > 0 else 0.0

    road_iou = get_iou(metrics[0]['TP'], metrics[0]['FP'], metrics[0]['FN'])
    farm_iou = get_iou(metrics[1]['TP'], metrics[1]['FP'], metrics[1]['FN'])
    mean_iou = (road_iou + farm_iou) / 2.0

    total_tp = metrics[0]['TP'] + metrics[1]['TP']
    total_fp = metrics[0]['FP'] + metrics[1]['FP']
    total_fn = metrics[0]['FN'] + metrics[1]['FN']

    val_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    val_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0

    return losses.avg, val_precision, val_recall, val_f1, road_iou, farm_iou, mean_iou


if __name__ == '__main__':
    os.makedirs(args.savedir, exist_ok=True)
    # running_file
    running_file_path = os.path.join(args.savedir,
                                     '%s_running-%s.txt' % (args.model, time.strftime('%Y-%m-%d-%H-%M-%S')))

    with open(running_file_path, 'w') as f:
        main(f)
    print('done')