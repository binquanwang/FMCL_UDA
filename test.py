from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.datasets.domain_adaptation import DA

from reid import models
from reid.trainers import Trainer
from reid.utils.data.sampler import RandomIdentitySampler
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, UnsupervisedCamStylePreprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint,copy_state_dict
from reid.loss import InvNet

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_data(data_dir, source, target, height, width, batch_size, re=0, workers=8):

    dataset = DA(data_dir, source, target)
    dataset_2 = DA(data_dir, target, source)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    num_classes = dataset.num_train_ids

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
        normalizer,
       # T.RandomErasing(EPSILON=re),
        T.RandomErasing(probability=0.4, mean=[0.485, 0.456, 0.406])
    ])

    train_transformer_2 = T.Compose([
        T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
        normalizer,
       # T.RandomErasing(EPSILON=re),
        T.RandomErasing(probability=0.4, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer,
    ])

    '''
    num_instances=4

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomIdentitySampler(dataset.target_train, num_instances)
    else:
        sampler = None
    '''
    source_train_loader = DataLoader(
        Preprocessor(dataset.source_train, root=osp.join(dataset.source_images_dir, dataset.source_train_path),
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    num_instances=0

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomIdentitySampler(dataset.target_train, num_instances)
    else:
        sampler = None



    target_train_loader = DataLoader(
        UnsupervisedCamStylePreprocessor(dataset.target_train,
                                         root=osp.join(dataset.target_images_dir, dataset.target_train_path),
                                         camstyle_root=osp.join(dataset.target_images_dir,
                                                                dataset.target_train_camstyle_path),
                                         num_cam=dataset.target_num_cam, transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True, drop_last=True)

    query_loader = DataLoader(
        Preprocessor(dataset.query,
                     root=osp.join(dataset.target_images_dir, dataset.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    query_loader_2 = DataLoader(
        Preprocessor(dataset_2.query,
                     root=osp.join(dataset_2.target_images_dir, dataset_2.query_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader = DataLoader(
        Preprocessor(dataset.gallery,
                     root=osp.join(dataset.target_images_dir, dataset.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    gallery_loader_2 = DataLoader(
        Preprocessor(dataset_2.gallery,
                     root=osp.join(dataset_2.target_images_dir, dataset_2.gallery_path), transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset,dataset_2, num_classes, source_train_loader, target_train_loader, query_loader, gallery_loader, query_loader_2, gallery_loader_2


def main(args):
    # For fast training.
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print('log_dir=', args.logs_dir)

    # Print logs
    print(args)

    # Create data loaders
    dataset,dataset_2, num_classes, source_train_loader, target_train_loader, \
    query_loader, gallery_loader,query_loader_2, gallery_loader_2 = get_data(args.data_dir, args.source,
                                            args.target, args.height,
                                            args.width, args.batch_size,
                                            args.re, args.workers)

    # Create model
    model = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)

    model_ema = models.create(args.arch, num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes)  #####new add

    # Invariance learning model
    num_tgt = len(dataset.target_train)
    model_inv = InvNet(args.features, num_tgt,
                        beta=args.inv_beta, knn=args.knn,
                        alpha=args.inv_alpha)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
      #  model_inv.load_state_dict(checkpoint['state_dict_inv'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))

    
    checkpoint = load_checkpoint(args.init_1)
    model.load_state_dict(checkpoint['state_dict'])
    model_ema.load_state_dict(checkpoint['state_dict'])
    
    '''    
    checkpoint = load_checkpoint(args.init_1)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model_ema.load_state_dict(model_dict)
    '''
    # Set model
    model = nn.DataParallel(model).to(device)
    model_ema = nn.DataParallel(model_ema).to(device)#####new add
    model_inv = model_inv.to(device)

    
    
    '''
    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model)
    copy_state_dict(initial_weights['state_dict'], model_ema)
    model_ema.module.classifier.weight.data.copy_(model.module.classifier.weight.data)
    '''
   # model.load_state_dict(checkpoint['state_dict'])
   # model_ema.load_state_dict(checkpoint['state_dict'])
   # copy_state_dict(initial_weights['state_dict'], model)
    #copy_state_dict(initial_weights['state_dict'], model_ema)
   # model_ema.module.classifier.weight.data.copy_(model.module.classifier.weight.data)
    
    # Evaluator
    evaluator = Evaluator(model)


    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, dataset.query,
                       dataset.gallery, args.output_feature)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Invariance Learning for Domain Adaptive Re-ID")
    # source
    parser.add_argument('-s', '--source', type=str, default='duke',
                        choices=['market', 'duke', 'msmt17'])
    # target
    parser.add_argument('-t', '--target', type=str, default='market',
                        choices=['market', 'duke', 'msmt17'])
    # imgs setting
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=4096)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for ImageNet pretrained"
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epochs_decay', type=int, default=30)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/data/ustc/wbq/data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    # random erasing
    parser.add_argument('--re', type=float, default=0.5)

    parser.add_argument('--init-1', type=str, default='/data/ustc/wbq/ECN-contrast/logs/duke2market-ECN/checkpoint.pth.tar', metavar='PATH')

    # Invariance learning
    parser.add_argument('--inv-alpha', type=float, default=0.4,
                        help='update rate for the exemplar memory in invariance learning')
    parser.add_argument('--inv-beta', type=float, default=0.07,
                        help='The temperature in invariance learning')
    parser.add_argument('--knn', default=18, type=int,
                        help='number of KNN for neighborhood invariance')
    parser.add_argument('--lmd', type=float, default=0.3,
                        help='weight controls the importance of the source loss and the target loss.')
    args = parser.parse_args()
    main(args)
