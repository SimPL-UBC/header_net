import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
import torch.utils.data
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import sys

# Add paths to import from NFL codebase
sys.path.append('../1st_place_kaggle_player_contact_detection/cnn/models')
sys.path.append('../1st_place_kaggle_player_contact_detection/cnn')
sys.path.append('./dataset')
sys.path.append('./configs')

from resnet3d_csn import ResNet3dCSN
from dataset_header_single import HeaderDataset, get_header_transforms
from header_default import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Header Detection Training')
    
    # Dataset parameters
    parser.add_argument('--train_list', type=str, required=True,
                        help='Path to training list file')
    parser.add_argument('--val_list', type=str, required=True,
                        help='Path to validation list file')
    parser.add_argument('--root_path', type=str, default='',
                        help='Root path for dataset')
    parser.add_argument('--store_name', type=str, default='header_net',
                        help='Name for storing model')
    
    # Model parameters
    parser.add_argument('--arch', type=str, default='resnet50',
                        help='Model architecture')
    parser.add_argument('--num_segments', type=int, default=11,
                        help='Number of temporal segments')
    parser.add_argument('--consensus_type', type=str, default='avg',
                        help='Consensus method')
    parser.add_argument('--k', type=int, default=3,
                        help='Top-k accuracy')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of total epochs to run')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--lr_type', type=str, default='step',
                        help='Learning rate decay type')
    parser.add_argument('--lr_steps', nargs='+', type=int, default=[20, 40],
                        help='Epochs to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Other parameters
    parser.add_argument('--modality', type=str, default='RGB',
                        help='Input modality')
    parser.add_argument('--dense_sample', action='store_true',
                        help='Use dense sampling')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='Print frequency')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='Evaluation frequency')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to latest checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on validation set')
    parser.add_argument('--snapshot_pref', type=str, default='',
                        help='Snapshot prefix')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='Manual epoch number')
    parser.add_argument('--gpus', nargs='+', type=int, default=None,
                        help='GPU ids to use')
    parser.add_argument('--flow_prefix', type=str, default='')
    parser.add_argument('--root_log', type=str, default='log',
                        help='Log directory')
    parser.add_argument('--root_model', type=str, default='checkpoint',
                        help='Model checkpoint directory')
    
    return parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        if args.gpus is not None:
            inputs = inputs.cuda(args.gpus[0], non_blocking=True)
            targets = targets.cuda(args.gpus[0], non_blocking=True)
        
        # Compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Measure accuracy and record loss
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        
        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(args, 'clip_gradient'):
            clip_grad_norm_(model.parameters(), args.clip_gradient)
        
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    # Switch to evaluate mode
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            if args.gpus is not None:
                inputs = inputs.cuda(args.gpus[0], non_blocking=True)
                targets = targets.cuda(args.gpus[0], non_blocking=True)
            
            # Compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
    
    # Calculate additional metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted'
    )
    
    # AUC (for binary classification)
    if len(np.unique(all_targets)) == 2:
        auc = roc_auc_score(all_targets, all_probs[:, 1])
    else:
        auc = 0.0
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f' * Acc@1 {top1.avg:.3f} Precision {precision:.3f} '
          f'Recall {recall:.3f} F1 {f1:.3f} AUC {auc:.3f}')
    print(f'Confusion Matrix:\n{cm}')
    
    return top1.avg, precision, recall, f1, auc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps, lr, gamma=0.1):
    """Sets the learning rate to the initial LR decayed by gamma every lr_steps epochs"""
    if lr_type == 'step':
        decay = gamma ** (sum(epoch >= np.array(lr_steps)))
        lr = lr * decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif lr_type == 'cos':
        import math
        lr = 0.5 * lr * (1 + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.root_log, exist_ok=True)
    os.makedirs(args.root_model, exist_ok=True)
    
    # Setup model
    print(f"Building model: {args.arch}")
    
    if args.arch == 'resnet50':
        model = ResNet3dCSN(
            pretrained2d=False,
            pretrained=None,
            depth=50,
            with_pool2=False,
            bottleneck_mode='ir',
            norm_eval=False,
            zero_init_residual=False,
            bn_frozen=False,
            num_classes=2  # Binary classification: header vs non-header
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    
    # Setup loss function with class weights
    class_weights = torch.FloatTensor([CLASS_WEIGHTS[0], CLASS_WEIGHTS[1]])
    if args.gpus is not None:
        class_weights = class_weights.cuda(args.gpus[0])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Setup GPU
    if args.gpus is not None:
        torch.cuda.set_device(args.gpus[0])
        model = model.cuda(args.gpus[0])
        if len(args.gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpus)
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    cudnn.benchmark = True
    
    # Data loading
    train_transform = get_header_transforms(input_size=224, is_training=True)
    val_transform = get_header_transforms(input_size=224, is_training=False)
    
    train_dataset = HeaderDataset(
        args.train_list,
        num_segments=args.num_segments,
        modality=args.modality,
        transform=train_transform,
        dense_sample=args.dense_sample,
        test_mode=False
    )
    
    val_dataset = HeaderDataset(
        args.val_list,
        num_segments=args.num_segments,
        modality=args.modality,
        transform=val_transform,
        dense_sample=args.dense_sample,
        test_mode=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    best_acc1 = 0
    best_f1 = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps, args.lr)
        print(f'Epoch: {epoch}, LR: {lr}')
        
        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        # Evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            acc1, precision, recall, f1, auc = validate(val_loader, model, criterion, args)
            
            # Remember best acc@1 and save checkpoint
            is_best = f1 > best_f1  # Use F1 score for best model selection
            best_acc1 = max(acc1, best_acc1)
            best_f1 = max(f1, best_f1)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=os.path.join(args.root_model, f'{args.store_name}_epoch_{epoch}.pth.tar'))
    
    print(f'Best Acc@1: {best_acc1:.3f}, Best F1: {best_f1:.3f}')

if __name__ == '__main__':
    main()
