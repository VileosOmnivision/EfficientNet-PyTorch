"""
Evaluate on ImageNet. Note that at the moment, training is not implemented (I am working on it).
that being said, evaluation is working.
"""

import argparse
import os
import random
import shutil
import time
import warnings
import logging
from datetime import datetime

import numpy as np
import PIL
from matplotlib import pyplot as plt
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim.lr_scheduler  # Add this import

from export_onnx import FotorrojoNet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--image_size', default=224, type=int,
                    help='image size')
parser.add_argument('--advprop', default=False, action='store_true',
                    help='use advprop or not')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
MODEL_PATH = r"C:\git\EfficientNet-PyTorch\fotorrojoNet"
OUTPUT_PATH = MODEL_PATH + "/results"

class FakeArgs:
    def __init__(self):
        self.data = "path/to/dataset"
        self.arch = "fotorrojoNet"  # Changed for FotorrojoNet
        self.workers = 4
        self.epochs = 90
        self.start_epoch = 0
        self.batch_size = 256
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-3
        self.print_freq = 10
        self.resume = ""
        self.evaluate = False
        self.pretrained = True # This might not be applicable to FotorrojoNet unless you load weights
        self.world_size = -1
        self.rank = -1
        self.dist_url = "tcp://224.66.41.62:23456"
        self.dist_backend = "nccl"
        self.seed = None
        self.gpu = 0
        self.image_size = (75, 225) # Changed for FotorrojoNet input HxW
        self.advprop = False
        self.multiprocessing_distributed = False

def main():
    # Create training session name with datetime
    training_session_name = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        args = parser.parse_args()
    except:
        # If no arguments are passed, use hardcoded defaults
        args = FakeArgs()
        args.data = "C:/datasets/fotorrojo/dataset_margen_alrededor"
        args.arch = "fotorrojoNet" # Explicitly set for FotorrojoNet
        args.workers = 8
        args.epochs = 80
        args.lr = 5e-4
        args.image_size = (75, 225) # Explicitly set for FotorrojoNet HxW
        args.batch_size = 128 # User specified
        args.resume = "" # "C:/git/EfficientNet-PyTorch/results/model_best_triple.pth.tar"

    # Add training session name to args for use in main_worker
    args.training_session_name = training_session_name

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def early_stopping(val_losses, patience=15, min_delta=0.001):
    """
    Check if training should be stopped based on validation loss
    """
    if len(val_losses) < patience + 1:
        return False

    # Check if validation loss has not improved for 'patience' epochs
    recent_losses = val_losses[-(patience+1):]
    best_loss = min(recent_losses[:-1])
    current_loss = recent_losses[-1]

    if current_loss > best_loss - min_delta:
        return True
    return False

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # Setup logging
    log_dir = os.path.join(OUTPUT_PATH, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Use training session name for all output files
    log_file = os.path.join(log_dir, f"{args.training_session_name}_training.log")
    csv_log_file = os.path.join(log_dir, f"{args.training_session_name}_training_metrics.csv")

    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Create CSV file for structured logging
    with open(csv_log_file, 'w', newline='') as f:
        f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy,learning_rate,is_best\n")

    logging.info("=" * 80)
    logging.info("TRAINING SESSION STARTED")
    logging.info("=" * 80)
    logging.info("Training Configuration:")
    logging.info("-" * 40)

    # Log all training arguments/configuration
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    logging.info("-" * 40)
    logging.info(f"Log files: {log_file}")
    logging.info(f"CSV metrics file: {csv_log_file}")
    logging.info("=" * 80)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        logging.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = len(os.listdir(os.path.join(args.data, 'train')))
    model = FotorrojoNet(num_classes=num_classes, input_size=args.image_size)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # Removed AlexNet/VGG specific logic, direct DataParallel for FotorrojoNet
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Add the learning rate scheduler
    # Cada 'step_size' epochs, el learning rate se multiplica por 'gamma'
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = 0  # or checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            # La siguiente línea carga el LR con el que se obtuvieron los pesos del entrenamiento del checkpoint
            # optimizer.load_state_dict(checkpoint['optimizer']) # <-- La comento porque empezaba con un LR súper bajo
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    # if 'efficientnet' in args.arch:
    #     image_size = EfficientNet.get_image_size(args.arch)
    # else:
    print("He cambiado esto para que me deje ponerle un tamaño de imagen menor de 224")
    # image_size will now be a tuple e.g. (25, 75) from args
    image_size = args.image_size

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC), # Resize to FotorrojoNet's input dimensions
            transforms.RandomApply([
                transforms.RandomCrop((50, 150), padding=0, pad_if_needed=False),  # random position crop to 50x150
                transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC), # Resize to FotorrojoNet's input dimensions
            ], p=0.5),  # solo se aplica un 50% de las veces
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=(0.8, 1.2),
                                   contrast=(0.8, 1.2)
                                   ),
            transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1)),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        #transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),  # Not good for traffic lights
        #transforms.CenterCrop(image_size),  # Not good for traffic lights
        transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC), # Resize to FotorrojoNet's input dimensions
        transforms.ToTensor(),
        normalize,
    ])
    print('Using image size', image_size)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        res = validate(val_loader, model, criterion, args)
        with open('res.txt', 'w') as f:
            print(res, file=f)
        return

    # Initialize lists for early stopping
    val_losses = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Log epoch start
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs} - Learning rate: {current_lr:.6f}")

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, criterion, args)

        # Add to early stopping monitoring
        val_losses.append(val_loss)

        # Log epoch results
        logging.info(f"Epoch {epoch + 1}/{args.epochs} completed:")
        logging.info(f"  Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        logging.info(f"  Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        logging.info(f"  Best Validation Accuracy so far: {max(val_acc, best_acc1):.2f}%")

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        # Save epoch metrics to CSV file
        with open(csv_log_file, 'a', newline='') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f},{current_lr:.6f},{is_best}\n")

        # Save checkpoint every 10 epochs
        if (epoch % 10 == 0 or is_best) and epoch > 0:
            print(f"Saving checkpoint at epoch {epoch}. Is best: {is_best}")
            logging.info(f"Saving checkpoint at epoch {epoch + 1}. Is best: {is_best}")
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.training_session_name)

        # Check for early stopping (only after epoch 20 to allow initial learning)
        # if epoch > 20 and early_stopping(val_losses, patience=12):
        #     logging.info(f"Early stopping triggered at epoch {epoch+1}")
        #     logging.info(f"Validation loss hasn't improved for 12 epochs")
        #     break

        # Step the scheduler
        scheduler.step()

    logging.info(f"Training completed! Final best accuracy: {best_acc1:.2f}%")

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, prefix="Epoch: [{}]".format(epoch))

    # Removed layer freezing logic to train the entire FotorrojoNet
    # If you want to freeze specific layers, you'll need to adapt the logic
    # for FotorrojoNet's architecture (e.g., model.conv_block1, model.fc_block)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        #top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    # Print training accuracy
    print(f"Training Accuracy: {top1.avg:.3f}")

    # Return loss and accuracy for logging
    return losses.avg, top1.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # Initialize list to store predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            #top5.update(acc5[0], images.size(0))

            # Store predictions and targets for confusion matrix
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    statistics_calc(OUTPUT_PATH, all_preds, all_targets, args.training_session_name)

    # Return loss and accuracy for logging
    return losses.avg, top1.avg

def save_checkpoint(state, is_best, training_session_name, filename='checkpoint.pth.tar'):
    # Use training session name for checkpoint files
    checkpoint_filename = f"{training_session_name}_checkpoint.pth.tar"
    best_filename = f"{training_session_name}_model_best.pth.tar"

    torch.save(state, os.path.join(OUTPUT_PATH, checkpoint_filename))
    if is_best:
        torch.save(state, os.path.join(OUTPUT_PATH, best_filename))


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


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Obsoleto, utilizo la función oficial
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(res) == 1:
            res = res[0]
        return res

def plot_roc(output_path, roc_auc, true_positive_rate, false_positive_rate, training_session_name):
    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % roc_auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # Save with training session name and datetime in filename
    plt.savefig(f"{output_path}/{training_session_name}_roc.jpg")
    #plt.show()

def statistics_calc(output_path, y_pred, y_true, training_session_name):
    # Confusion matrix
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    class_report = metrics.classification_report(y_true, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print(class_report)

    # ROC & AUC
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(y_true, y_pred)
    plot_roc(output_path, roc_auc, true_positive_rate, false_positive_rate, training_session_name)


if __name__ == '__main__':
    main()
