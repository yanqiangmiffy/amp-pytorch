#!/usr/bin/env python
# -*- coding:utf-8 _*-
""" 
@author:quincy qiang
@license: Apache Licence
@file: main.py
@time: 2022/04/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T
import argparse
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from models import ResNet50,ResNet18
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))  # 获取当前路径


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


def plot_learning_curves(metrics, cur_epoch, args):
    x = np.arange(cur_epoch + 1)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ln1 = ax1.plot(x, metrics['train_loss'], color='tab:red')
    ln2 = ax1.plot(x, metrics['val_loss'], color='tab:red', linestyle='dashed')
    ax1.grid()
    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ln3 = ax2.plot(x, metrics['train_acc'], color='tab:blue')
    ln4 = ax2.plot(x, metrics['val_acc'], color='tab:blue', linestyle='dashed')
    lns = ln1 + ln2 + ln3 + ln4
    plt.legend(lns, ['Train loss', 'Validation loss', 'Train accuracy', 'Validation accuracy'])
    plt.tight_layout()
    plt.savefig('{}/{}/learning_curve.png'.format(args.checkpoint_dir, args.checkpoint_name), bbox_inches='tight')
    plt.close('all')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_args():
    parser = argparse.ArgumentParser()

    # model architecture & checkpoint
    parser.add_argument('--model', default='ResNet18', choices=('ResNet18', 'ResNet50'),
                        help='optimizer to use (ResNet18 | ResNet50)')
    parser.add_argument('--norm', default='batchnorm')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--pretrained_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_name', type=str, default='')

    # data loading
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--weight', type=int, default=224, help='random seed')
    parser.add_argument('--height', type=int, default=224, help='random seed')
    parser.add_argument('--num_chanel', type=int, default=3, help='random seed')

    # training hyper parameters
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False)

    # optimzier & learning rate scheduler
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM'),
                        help='optimizer to use (SGD | ADAM)')
    parser.add_argument('--decay_type', default='cosine_warmup', choices=('step', 'step_warmup', 'cosine_warmup'),
                        help='optimizer to use (step | step_warmup | cosine_warmup)')

    args = parser.parse_args()
    return args


args = get_args()
print(args)


def create_dataloader(args):
    train_trans = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    valid_trans = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    test_trans = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root="data/seg_train/seg_train", transform=train_trans)
    validset = torchvision.datasets.ImageFolder(root="data/seg_train/seg_train", transform=valid_trans)
    testset = torchvision.datasets.ImageFolder(root="data/seg_test/seg_test", transform=test_trans)

    np.random.seed(args.seed)
    targets = trainset.targets
    train_idx, valid_idx = train_test_split(np.arange(len(targets)), test_size=0.2, shuffle=True, stratify=targets)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers
    )

    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    return train_loader, valid_loader, test_loader


train_loader, valid_loader, test_loader = create_dataloader(args)

model = ResNet18(
    (args.weight, args.height, args.num_chanel),
    args.num_classes,
    checkpoint_dir=args.checkpoint_dir,
    checkpoint_name=args.checkpoint_name,
    pretrained=args.pretrained,
    pretrained_path=args.pretrained_path,
    norm=args.norm,
)
if torch.cuda.device_count() >= 1:
    print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
    model = model.cuda()
else:
    raise ValueError('CPU training is not supported')

# 训练设置
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=args.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
""" define loss scaler for automatic mixed precision 
在混合精度计算中使用float16数据格式数据动态范围降低，造成梯度计算出现浮点溢出，会导致部分参数更新失败。为了保证部分模型训练在混合精度训练过程中收敛，需要配置Loss Scaling的方法。
Loss Scaling方法通过在前向计算所得的loss乘以loss scale系数S，起到在反向梯度计算过程中达到放大梯度的作用，从而最大程度规避浮点计算中较小梯度值无法用FP16表达而出现的溢出问题。
在参数梯度聚合之后以及优化器更新参数之前，将聚合后的参数梯度值除以loss scale系数S还原。
"""
scaler = torch.cuda.amp.GradScaler()

# 训练记录
result_dict = {'args': vars(args), 'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'test_acc': []}
train_time_list = []
valid_time_list = []
best_val_acc = 0.0
""" 每轮迭代训练和评估"""


def train(model, data_loader, epoch, args, result_dict):
    total_loss = 0
    count = 0
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.cuda(), labels.cuda()

        if args.amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if len(labels.size()) > 1:
            labels = torch.argmax(labels, axis=1)

        prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

        optimizer.zero_grad()

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.tolist()
        count += labels.size(0)

        if batch_idx % args.log_interval == 0:
            _s = str(len(str(len(data_loader.sampler))))
            ret = [
                ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.sampler),
                                                                           100 * count / len(data_loader.sampler)),
                'train_loss: {: >4.2e}'.format(total_loss / count),
                'train_accuracy : {:.2f}%'.format(top1.avg)
            ]
            print(', '.join(ret))

    scheduler.step()
    result_dict['train_loss'].append(losses.avg)
    result_dict['train_acc'].append(top1.avg)
    return result_dict


def evaluate(model, data_loader, epoch, args, result_dict):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            if args.amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

    print('----Validation Results Summary----')
    print('Epoch: [{}] Top-1 accuracy: {:.2f}%'.format(epoch, top1.avg))

    result_dict['val_loss'].append(losses.avg)
    result_dict['val_acc'].append(top1.avg)
    return result_dict


def predict(data_loader, args, result_dict):
    top1 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            top1.update(prec1.item(), inputs.size(0))

    print('----Test Set Results Summary----')
    print('Top-1 accuracy: {:.2f}%'.format(top1.avg))

    result_dict['test_acc'].append(top1.avg)

    return result_dict


for epoch in range(args.epochs):

    result_dict['epoch'] = epoch
    torch.cuda.synchronize()

    # 训练开始
    tic1 = time.time()
    result_dict = train(model, train_loader, epoch, args, result_dict)

    torch.cuda.synchronize()
    tic2 = time.time()
    train_time_list.append(tic2 - tic1)
    # 训练结束

    # 评估开始
    torch.cuda.synchronize()
    tic3 = time.time()
    result_dict = evaluate(model, valid_loader, epoch, args, result_dict)

    torch.cuda.synchronize()
    tic4 = time.time()
    valid_time_list.append(tic4 - tic3)
    # 评估结束
    checkpoint_name = 'best_model'
    if result_dict['val_acc'][-1] > best_val_acc:
        print("{} epoch, best epoch was updated! {}%".format(epoch, result_dict['val_acc'][-1]))
        best_val_acc = result_dict['val_acc'][-1]
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name, checkpoint_name + '.pt')
        torch.save(model.state_dict(), checkpoint_path)

    save_path = os.path.join(args.checkpoint_dir, args.checkpoint_name, 'result_dict.json')
    with open(save_path, 'w') as f:
        f.write(json.dumps(result_dict, sort_keys=True, indent=4, ensure_ascii=False))
    plot_learning_curves(result_dict, epoch, args)

    # # 测试集评估 best model
    # checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name, checkpoint_name + '.pt')
    # model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    # result_dict = evaluator.test(test_loader, args, result_dict)
    # evaluator.save(result_dict)
    np.savetxt(os.path.join(model.checkpoint_dir, model.checkpoint_name, 'train_time_amp.csv'), train_time_list, delimiter=',', fmt='%s')
