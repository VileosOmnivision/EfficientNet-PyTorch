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
import torch.quantization
from torchvision.transforms import InterpolationMode

from export_onnx import FotorrojoNet, export_onnx_model

class RGBtoBGR(object):
    """Convert RGB image tensor to BGR format to match OpenCV's default"""
    def __call__(self, tensor):
        # tensor is (C, H, W) format from ToTensor()
        # swap channels: RGB -> BGR
        return tensor[[2, 1, 0], :, :]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
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
parser.add_argument('--sanity-test', dest='sanity_test', action='store_true',
                    help='Run sanity test on test images organized by class folders')
parser.add_argument('--test-data', default='', type=str, metavar='PATH',
                    help='Path to test dataset (default: data/test)')
parser.add_argument('--model-checkpoint', default='', type=str, metavar='PATH',
                    help='Path to model checkpoint for sanity testing')

best_acc1 = 0
MODEL_PATH = r"C:\git\EfficientNet-PyTorch\fotorrojoNet"

class FakeArgs:
    def __init__(self):
        # Training configuration
        self.short_name = "bilbao2"
        self.description = "Quantization Aware Training. 2321 images per class."
        self.data = "C:/datasets/fotorrojo/dataset_margen_alrededor"
        self.arch = "fotorrojoNet"
        self.workers = 8
        self.epochs = 45
        self.start_epoch = 0
        self.batch_size = 128
        self.lr = 3e-3
        self.momentum = 0.7
        self.weight_decay = 5e-4
        self.print_freq = 10
        self.resume = ""
        self.evaluate = False
        self.pretrained = False # This might not be applicable to FotorrojoNet unless you load weights
        self.world_size = -1
        self.rank = -1
        self.dist_url = "tcp://224.66.41.62:23456"
        self.dist_backend = "nccl"
        self.seed = None
        self.gpu = 0
        self.image_size = (75, 225)
        self.advprop = False
        self.multiprocessing_distributed = False
        self.early_stopping = False
        self.early_stopping_patience = 10
        self.qat = True # Enable Quantization-Aware Training
        self.qat_epoch = 3 # Epoch to start converting model to quantized version

        # Learning rate scheduler parameters
        self.scheduler_factor = 0.7      # Factor to reduce LR by
        self.scheduler_patience = 10     # Epochs to wait before reducing LR
        self.scheduler_mode = 'min'      # Monitor validation loss decrease

        # Sanity test arguments
        self.sanity_test = False
        self.test_data = r"C:\datasets\fotorrojo\ayto_Madrid_dic2024"  # Will default to data/test if empty
        self.sanity_model_weights = r""  # Will auto-find latest if empty

def main(other_dataset=None):
    # Create training session name with datetime
    session_name = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        args = parser.parse_args()
    except:
        # If no arguments are passed, use hardcoded defaults
        print("No command line arguments detected. Using hardcoded defaults for debugging.")
        args = FakeArgs()
        if other_dataset is not None:
            args.data = other_dataset
    # Add training session name to args for use in main_worker
    args.session_name = f"{session_name}_{args.short_name}"

    # Handle sanity test mode
    if args.sanity_test:
        # Set default values if using FakeArgs
        if not args.test_data:
            args.test_data = os.path.join(args.data, 'val')
        if not args.sanity_model_weights:
            # Try to find the most recent model checkpoint
            training_history_path = os.path.join(MODEL_PATH, "training_history")
            if os.path.exists(training_history_path):
                # Find the most recent session folder
                session_folders = [d for d in os.listdir(training_history_path)
                                    if os.path.isdir(os.path.join(training_history_path, d))]
                if session_folders:
                    # Sort session folders by modification time (most recent last)
                    session_folders.sort(key=lambda d: os.path.getmtime(os.path.join(training_history_path, d)))
                    print(session_folders)
                    latest_session = session_folders[-1]
                    args.sanity_model_weights = os.path.join(training_history_path, latest_session, f"{latest_session}_model_best.pth.tar")
                    print(f"Using latest model checkpoint: {args.sanity_model_weights}")
                else:
                    print(f"No session folders found in {training_history_path}, cannot run sanity test without model weights")
            else:
                print(f"No training history found at {training_history_path}, cannot run sanity test without model weights")

        # Run sanity test and exit
        success = sanity_test(args)
        if not success:
            exit(1)
        return

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
        print("Early stopping check: not enough data")
        return False

    # Check if validation loss has not improved for 'patience' epochs
    recent_losses = val_losses[-(patience+1):]
    best_loss = min(recent_losses[:-1])
    current_loss = recent_losses[-1]

    if current_loss > best_loss - min_delta:
        return True
    print(f"Early stopping check: current_loss={current_loss:.4f}, best_loss={best_loss:.4f}")
    return False

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # Create output directory structure: training_history/session_name_short_description/
    output_path = os.path.join(MODEL_PATH, "training_history", args.session_name)
    os.makedirs(output_path, exist_ok=True)

    # Setup logging - all files go directly in the session folder
    log_file = os.path.join(output_path, f"{args.session_name}_training.log")
    csv_log_file = os.path.join(output_path, f"{args.session_name}_training_metrics.csv")

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
        f.write("epoch,train_loss,train_accuracy,val_loss,val_accuracy,val_f1_score,learning_rate,is_best\n")

    logging.info("=" * 80)
    logging.info("TRAINING SESSION STARTED")
    logging.info("=" * 80)
    logging.info(f"Session Name: {args.session_name}")
    if hasattr(args, 'description') and args.description:
        logging.info(f"Description: {args.description}")
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

    if args.qat:
        print("Preparing model for Quantization-Aware Training (QAT)")
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)

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

    # define loss function (criterion) and optimizer (will be updated with class weights later)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Add the learning rate scheduler - Using configurable parameters
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=args.scheduler_mode,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        verbose=True
    )

    # optionally resume from a checkpoint
    if args.resume == 'last':
        # Try to find the most recent model checkpoint
        training_history_path = os.path.join(MODEL_PATH, "training_history")
        if os.path.exists(training_history_path):
            # Find the most recent session folder
            session_folders = [d for d in os.listdir(training_history_path)
                                if os.path.isdir(os.path.join(training_history_path, d))]
            if session_folders:
                # Sort session folders by modification time (most recent last)
                session_folders.sort(key=lambda d: os.path.getmtime(os.path.join(training_history_path, d)))
                latest_session = session_folders[-2]
                args.resume = os.path.join(training_history_path, latest_session, f"{latest_session}_checkpoint.pth.tar")
                print(f"Using latest model checkpoint: {args.resume}")
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
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.RandomCrop((60, 180), padding=0, pad_if_needed=False),
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            ], p=0.4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RGBtoBGR(),  # Convert RGB to BGR to match OpenCV format used in RKNN inference
            # transforms.ColorJitter(brightness=(0.8, 1.2),
            #                        contrast=(0.8, 1.2)
            #                        ),
            # transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1)),
            # normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_transforms = transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC), # Fixed interpolation warning
        transforms.ToTensor(),
        RGBtoBGR(),  # Convert RGB to BGR to match OpenCV format used in RKNN inference
        # normalize,
    ])
    print('Using image size', image_size)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Calculate class weights for imbalanced dataset after data loaders are created
    print("Calculating class distribution for balanced training...")
    class_counts = torch.zeros(num_classes)
    for _, target in train_loader:
        for t in target:
            class_counts[t] += 1

    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.cuda(args.gpu) if args.gpu is not None else class_weights

    print(f"Number of classes detected: {num_classes}")
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Update criterion with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)

    # Ensure model parameters use float32 precision
    model.float()

    if args.evaluate:
        res = validate(val_loader, model, criterion, args, output_path)
        with open(os.path.join(output_path, 'res.txt'), 'w') as f:
            print(res, file=f)
        return

    # Initialize lists for early stopping
    val_losses = []

    # Initialize lists for storing training metrics for plotting
    epoch_numbers = []
    train_losses = []
    train_accuracies = []
    val_losses_plot = []
    val_accuracies = []
    val_f1_scores = []
    learning_rates = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.qat and epoch > args.qat_epoch:
            # Freeze quantization parameters after a few epochs
            model.apply(torch.quantization.disable_observer)
        if args.qat and epoch > args.qat_epoch + 1:
            # Freeze batch norm stats
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Log epoch start
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Starting epoch {epoch + 1}/{args.epochs} - Learning rate: {current_lr:.6f}")

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, val_acc, val_f1 = validate(val_loader, model, criterion, args, output_path)

        # Add to early stopping monitoring
        val_losses.append(val_loss)

        # Store metrics for plotting
        epoch_numbers.append(epoch + 1)
        train_losses.append(float(train_loss))
        train_accuracies.append(float(train_acc))
        val_losses_plot.append(float(val_loss))
        val_accuracies.append(float(val_acc))
        val_f1_scores.append(float(val_f1))
        learning_rates.append(float(current_lr))

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
            f.write(f"{epoch + 1},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f},{val_f1:.4f},{current_lr:.6f},{is_best}\n")

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
            }, is_best, args.session_name, output_path)

        # Export ONNX model every 15 epochs
        if epoch % 15 == 0 and epoch > 0:
            print(f"Exporting ONNX model at epoch {epoch}")
            logging.info(f"Exporting ONNX model at epoch {epoch + 1}")

            # Create checkpoint filename for current epoch
            epoch_checkpoint_name = f"{args.session_name}_checkpoint.pth.tar"
            epoch_checkpoint_path = os.path.join(output_path, epoch_checkpoint_name)

            # Export ONNX with epoch suffix
            success = export_onnx_model(
                checkpoint_path=epoch_checkpoint_path,
                output_dir=output_path,
                session_name=f"{args.session_name}_epoch_{epoch + 1}",
                num_classes=len(os.listdir(os.path.join(args.data, 'train'))),
                input_size=args.image_size
            )

            if success:
                logging.info(f"ONNX export for epoch {epoch + 1} completed successfully!")
            else:
                logging.info(f"ONNX export for epoch {epoch + 1} failed!")

        # Check for early stopping (only after epoch 20 to allow initial learning)
        if args.early_stopping and epoch > 20 and early_stopping(val_losses, patience=args.early_stopping_patience):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            logging.info(f"Validation loss hasn't improved for {args.early_stopping_patience} epochs")
            break

        # Step the scheduler with validation loss (required for ReduceLROnPlateau)
        scheduler.step(val_loss)

    logging.info(f"Training completed! Final best accuracy: {best_acc1:.2f}%")

    # Plot training metrics evolution
    logging.info("=" * 80)
    logging.info("GENERATING TRAINING METRICS PLOTS")
    logging.info("=" * 80)

    if epoch_numbers:  # Only plot if we have data
        plot_training_metrics(
            epoch_numbers=epoch_numbers,
            train_losses=train_losses,
            train_accuracies=train_accuracies,
            val_losses=val_losses_plot,
            val_accuracies=val_accuracies,
            val_f1_scores=val_f1_scores,
            learning_rates=learning_rates,
            output_path=output_path,
            session_name=args.session_name
        )
        logging.info("Training metrics plots generated successfully!")
    else:
        logging.info("No training data to plot (training may have been skipped)")

    # Export trained model to ONNX format
    logging.info("=" * 80)
    logging.info("EXPORTING MODEL TO ONNX")
    logging.info("=" * 80)

    if args.qat:
        print("Converting QAT model to a quantized integer model for export")

        # Unwrap the model from DataParallel if necessary
        model_to_convert = model.module if isinstance(model, torch.nn.DataParallel) else model

        # Ensure quantization uses CPU kernels
        torch.backends.quantized.engine = 'fbgemm'
        torch.set_default_tensor_type(torch.FloatTensor)

        # Ensure model is in eval mode and move everything to CPU
        model_to_convert.eval()
        model_to_convert.to('cpu')

        # Move all parameters and buffers to CPU explicitly
        for param in model_to_convert.parameters():
            param.data = param.data.cpu()
        for buffer in model_to_convert.buffers():
            buffer.data = buffer.data.cpu()

        torch.quantization.convert(model_to_convert, inplace=True)

    # Find the best model checkpoint
    best_checkpoint_path = os.path.join(output_path, f"{args.session_name}_model_best.pth.tar")

    # Get number of classes from the model
    num_classes = len(os.listdir(os.path.join(args.data, 'train')))

    # Export the model
    success = export_onnx_model(
        checkpoint_path=best_checkpoint_path,
        output_dir=output_path,
        session_name=args.session_name,
        num_classes=num_classes,
        input_size=args.image_size
    )

    if success:
        logging.info("ONNX export completed successfully!")
    else:
        logging.info("ONNX export failed!")

    logging.info("=" * 80)

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

        # Debug: Print some output statistics
        if i == 0:  # Only print for first batch to avoid spam
            print(f"Debug - Output shape: {output.shape}")
            print(f"Debug - Output sample: {output[0].detach().cpu()}")
            print(f"Debug - Target sample: {target[:5].cpu()}")
            print(f"Debug - Loss: {loss.item():.4f}")

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

def validate(val_loader, model, criterion, args, output_path):
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

    # Calculate F1-score with zero_division parameter to avoid warnings
    f1 = metrics.f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    print(f' * F1-Score {f1:.3f}')

    statistics_calc(output_path, all_preds, all_targets, args.session_name)

    # Return loss, accuracy, and F1-score for logging
    return losses.avg, top1.avg, f1

def save_checkpoint(state, is_best, session_name, output_path, filename='checkpoint.pth.tar'):
    # Use training session name for checkpoint files
    checkpoint_filename = f"{session_name}_checkpoint.pth.tar"
    best_filename = f"{session_name}_model_best.pth.tar"

    torch.save(state, os.path.join(output_path, checkpoint_filename))
    if is_best:
        torch.save(state, os.path.join(output_path, best_filename))


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
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if len(res) == 1:
            res = res[0]
        return res

def plot_training_metrics(epoch_numbers, train_losses, train_accuracies, val_losses, val_accuracies, val_f1_scores, learning_rates, output_path, session_name):
    """Plot training metrics evolution over epochs"""

    # Convert all inputs to numpy arrays to ensure proper data types
    epoch_numbers = np.array(epoch_numbers, dtype=float)
    train_losses = np.array(train_losses, dtype=float)
    train_accuracies = np.array(train_accuracies, dtype=float)
    val_losses = np.array(val_losses, dtype=float)
    val_accuracies = np.array(val_accuracies, dtype=float)
    val_f1_scores = np.array(val_f1_scores, dtype=float)
    learning_rates = np.array(learning_rates, dtype=float)

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics Evolution - {session_name}', fontsize=16)

    # Plot 1: Loss curves
    ax1.plot(epoch_numbers, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epoch_numbers, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax2.plot(epoch_numbers, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epoch_numbers, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Rate
    ax3.plot(epoch_numbers, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')  # Log scale for learning rate
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: F1-Score evolution
    ax4.plot(epoch_numbers, val_f1_scores, 'purple', label='Validation F1-Score', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('F1-Score Evolution')
    ax4.set_ylim([0, 1])  # F1-score ranges from 0 to 1
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = os.path.join(output_path, f"{session_name}_training_metrics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close to free memory

    print(f"Training metrics plot saved to: {plot_path}")
    return plot_path

def plot_roc(output_path, roc_auc, true_positive_rate, false_positive_rate, session_name):
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
    plt.savefig(f"{output_path}/{session_name}_roc.jpg")
    #plt.show()

def statistics_calc(output_path, y_pred, y_true, session_name):
    # Confusion matrix
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    class_report = metrics.classification_report(y_true, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print(class_report)

    # ROC & AUC - Handle both binary and multi-class scenarios
    num_classes = len(np.unique(y_true))

    if num_classes == 2:
        # Binary classification - original behavior
        roc_auc = metrics.roc_auc_score(y_true, y_pred)
        false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_true, y_pred)
        plot_roc(output_path, roc_auc, true_positive_rate, false_positive_rate, session_name)
    else:
        # Multi-class classification - use One-vs-Rest approach
        print(f"Multi-class classification detected ({num_classes} classes)")
        print("ROC AUC calculation requires probability scores for multi-class, skipping ROC plot")
        print("Use classification report above for detailed metrics")

def sanity_test(args):
    """
    Run sanity test on test images organized by class folders
    The test dataset should be organized as:
    test_data/
        ├── class1/
        │   ├── image1.jpg
        │   └── image2.jpg
        └── class2/
            ├── image3.jpg
            └── image4.jpg
    """
    print("=" * 80)
    print("STARTING SANITY TEST")
    print("=" * 80)

    # Determine test data path
    if not args.test_data:
        args.test_data = os.path.join(args.data, 'test')

    if not os.path.exists(args.test_data):
        print(f"Error: Test data path '{args.test_data}' not found!")
        print("Please specify --test-data path or create a 'test' folder in your dataset directory")
        return False

    # Get class names from test directory
    class_names = sorted([d for d in os.listdir(args.test_data)
                         if os.path.isdir(os.path.join(args.test_data, d))])

    if not class_names:
        print(f"Error: No class folders found in '{args.test_data}'")
        return False

    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # Load model
    model = FotorrojoNet(num_classes=num_classes, input_size=args.image_size)

    # Load checkpoint
    if not args.sanity_model_weights:
        print("Error: Please specify --model-checkpoint path for sanity testing")
        return False

    if not os.path.exists(args.sanity_model_weights):
        print(f"Error: Model checkpoint '{args.sanity_model_weights}' not found!")
        return False

    print(f"Loading model checkpoint: {args.sanity_model_weights}")
    checkpoint = torch.load(args.sanity_model_weights, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # Filter out QAT-only observer buffers before loading weights
    model_state = model.state_dict()
    filtered_state = {k: v for k, v in new_state_dict.items() if k in model_state}

    dropped_keys = set(new_state_dict.keys()) - set(filtered_state.keys())
    if dropped_keys:
        print(f"Ignoring {len(dropped_keys)} quantization observer buffers during load")

    missing_keys = set(model_state.keys()) - set(filtered_state.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} model parameters missing from checkpoint")

    model.load_state_dict(filtered_state, strict=False)

    # Move model to GPU if available
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print("Model loaded successfully!")

    # Define test transforms (same as validation)
    if args.advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    test_transforms = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=InterpolationMode.BICUBIC), # Fixed interpolation warning
        transforms.ToTensor(),
        RGBtoBGR(),  # Convert RGB to BGR to match OpenCV format used in RKNN inference
        # normalize,  # Comment out if not used during training
    ])

    # Create test dataset
    test_dataset = datasets.ImageFolder(args.test_data, test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Test dataset loaded: {len(test_dataset)} images")

    # Run inference
    all_preds = []
    all_targets = []
    all_probs = []
    correct_predictions = 0
    total_predictions = 0

    print("\nRunning inference on test images...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            elif torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            # Forward pass
            outputs = model(images)

            # Get predictions
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

            # Calculate batch accuracy
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")

    # Calculate overall accuracy
    overall_accuracy = (correct_predictions / total_predictions) * 100

    print("=" * 80)
    print("SANITY TEST RESULTS")
    print("=" * 80)
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({correct_predictions}/{total_predictions})")

    # Per-class accuracy
    print("\nPer-class results:")
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(all_targets)):
        true_class = all_targets[i]
        pred_class = all_preds[i]

        class_total[true_class] += 1
        if true_class == pred_class:
            class_correct[true_class] += 1

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            accuracy = (class_correct[i] / class_total[i]) * 100
            print(f"  {class_name}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {class_name}: No test samples")

    # Detailed results for each image
    print("\nDetailed predictions:")
    print("-" * 60)

    # Get image paths from dataset
    image_paths = [test_dataset.samples[i][0] for i in range(len(test_dataset))]

    for i, (true_idx, pred_idx, probs) in enumerate(zip(all_targets, all_preds, all_probs)):
        true_class = class_names[true_idx]
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx] * 100
        is_correct = "✓" if true_idx == pred_idx else "✗"

        image_name = os.path.basename(image_paths[i])
        print(f"{is_correct} {image_name:<25} True: {true_class:<10} Pred: {pred_class:<10} ({confidence:.1f}%)")

    # Generate confusion matrix and classification report
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX AND CLASSIFICATION REPORT")
    print("=" * 80)

    conf_matrix = metrics.confusion_matrix(all_targets, all_preds)
    class_report = metrics.classification_report(all_targets, all_preds, target_names=class_names)

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Generate ROC curve if binary classification
    if num_classes == 2:
        try:
            # For binary classification, use probabilities of positive class
            positive_probs = [prob[1] for prob in all_probs]
            roc_auc = metrics.roc_auc_score(all_targets, positive_probs)
            false_positive_rate, true_positive_rate, _ = metrics.roc_curve(all_targets, positive_probs)

            print(f"\nROC AUC Score: {roc_auc:.4f}")

            # Save ROC plot if output path is available
            if hasattr(args, 'session_name'):
                output_path = os.path.join(MODEL_PATH, "training_history", args.session_name)
                if os.path.exists(output_path):
                    plot_roc(output_path, roc_auc, true_positive_rate, false_positive_rate, f"{args.session_name}_sanity_test")
                    print(f"ROC curve saved to: {output_path}/{args.session_name}_sanity_test_roc.jpg")
        except Exception as e:
            print(f"Could not generate ROC curve: {e}")

    print("=" * 80)
    print("SANITY TEST COMPLETED")
    print("=" * 80)

    return True


if __name__ == '__main__':
    main()
    # main("C:/datasets/fotorrojo/dataset_together_sep25")
