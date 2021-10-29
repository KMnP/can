#!/usr/bin/env python3
"""
Get imagenet logits and targets for training and val set
"""
import argparse
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tqdm import tqdm
from PIL import ImageFilter


def setup():
    """set up args parser"""
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(
        description='ImageNet Preparation')
    # Output:
    parser.add_argument('--out-dir', type=str)
    # Datasets
    parser.add_argument('--data-root', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--transform', default=None, type=int, help='additional image transformation applied, e.g. 32')
    # 2, 4, 6, 8, 16, 32
    parser.add_argument('--val-only', default=False, action='store_true',
                        help='if true, work on validation set only')

    # Architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')

    # Device options
    parser.add_argument('--gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    return args, use_cuda


def setup_transforms(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    if args.transform is not None:
        kernel = args.transform

        def _gaussian_blur(img):
            img_blur = img.filter(ImageFilter.GaussianBlur(kernel))
            return img_blur

        transform_list.append(transforms.Lambda(_gaussian_blur))
    transform_list.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(transform_list)


def data_setup(args):
    my_transforms = setup_transforms(args)
    valdir = os.path.join(args.data_root, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, my_transforms),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print("Validation set loaded (total = {})".format(len(val_loader.dataset)))

    if args.val_only:
        return None, val_loader

    start = time.time()
    traindir = os.path.join(args.data_root, 'train')
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, my_transforms),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    end = time.time()
    elapse = end - start
    print("Training set loaded (total = {}, took {:.2f} seconds)".format(
        len(train_loader.dataset), elapse))

    return train_loader, val_loader


def model_setup(args, use_cuda):
    print("Using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    if use_cuda:
        model = model.cuda()
    return model


def get_output(testloader, model, use_cuda):
    """ get output (after softmax opts) and target"""
    model.eval()

    output_list = []
    target_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            outputs = model(inputs)

            output_list.append(outputs.cpu())   # (batch_size, num_clses)
            target_list.append(targets.cpu())
            del inputs
            del outputs
            del targets
            torch.cuda.empty_cache()

    out = torch.cat(output_list, dim=0)      # num_image x num_cls
    targets = torch.cat(target_list, dim=0)  # num_image
    return torch.softmax(out, dim=1), targets


def main():
    print("=" * 80)
    print("preparing imagenet results")
    print("=" * 80)
    args, use_cuda = setup()
    model = model_setup(args, use_cuda)
    train_loader, val_loader = data_setup(args)

    print("Getting validation data...")
    logits, targets = get_output(val_loader, model, use_cuda)
    out_dict = {"logits": logits, "targets": targets}
    out_path = os.path.join(args.out_dir, "val.pth")
    torch.save(out_dict, out_path)
    print(f"==> Result saved at {out_path}")

    if not args.val_only:
        print("Getting training data...")
        logits, targets = get_output(train_loader, model, use_cuda)
        out_dict = {"logits": logits, "targets": targets}
        out_path = os.path.join(args.out_dir, "train.pth")
        torch.save(out_dict, out_path)
        print(f"==> Result saved at {out_path}")


if __name__ == '__main__':
    main()
