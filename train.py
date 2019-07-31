from __future__ import print_function, division

# https://github.com/yu4u/age-estimation-pytorch
# https://github.com/apachecn/pytorch-doc-zh
# https://github.com/verigak/progress

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import argparse
from pathlib2 import Path
# from tqdm import tqdm
# from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import time

from utils.helper import AverageMeter, resume_checkpoint, one_hot
from model import MultiTaskNet
from config import cfg
from dataset import MultiTaskDataset
from progress.bar import Bar
from visdom import Visdom


def get_args():
    parser = argparse.ArgumentParser(description='Images and corresponding Labels')
    parser.add_argument("--images_path", type=str, default='./data/images', help="Image root directory")
    parser.add_argument("--labels_path", type=str, default='./data/labels', help="Label root directory")
    parser.add_argument("--yaml", type=str, default='config/config.yaml', help='Yaml config path')
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--ratio", type=float, default=0.2, help="Train and validate data ratio")
    parser.add_argument("--tensorboard", type=str, required=True, help="Tensorboard log directory")
    args = parser.parse_args()
    return args


def train(train_dataloader, model, criterion, optimizer, epoch, device):
    """Training step"""
    model.train()

    # Smooth loss function
    face_loss = AverageMeter()
    mouth_loss = AverageMeter()
    eyebrow_loss = AverageMeter()
    eye_loss = AverageMeter()
    nose_loss = AverageMeter()
    jaw_loss = AverageMeter()
    total_loss = AverageMeter()

    face_acc = AverageMeter()
    mouth_acc = AverageMeter()
    eyebrow_acc = AverageMeter()
    eye_acc = AverageMeter()
    nose_acc = AverageMeter()
    jaw_acc = AverageMeter()
    total_acc = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()

    # with tqdm(train_dataloader) as _tqdm:
    #     for x, y0, y1, y2, y3, y4, y5 in _tqdm:
    bar = Bar('Processing train', max=len(train_dataloader))
    
    for batch_idx, (x, y0, y1, y2, y3, y4, y5) in enumerate(train_dataloader):
        data_time.update(time.time() - end)

        x = x.to(device)
        # y0 ~ y5 represent face, mouth, eyebrow, eye, nose, jaw
        y0 = y0.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        y3 = y3.to(device)
        y4 = y4.to(device)
        y5 = y5.to(device)
        outputs = model(x)
        sample_num = x.size(0)

        # Calculate loss
        # https://pytorch.org/docs/stable/nn.html?highlight=nn%20crossentropyloss#torch.nn.CrossEntropyLoss
        face_cur_loss = criterion(outputs[0], y0)
        face_cur_loss_ = face_cur_loss.item()
        mouth_cur_loss = criterion(outputs[1], y1)
        mouth_cur_loss_ = mouth_cur_loss.item()
        eyebrow_cur_loss = criterion(outputs[2], y2)
        eyebrow_cur_loss_ = eyebrow_cur_loss.item()
        eye_cur_loss = criterion(outputs[3], y3)
        eye_cur_loss_ = eye_cur_loss.item()
        nose_cur_loss = criterion(outputs[4], y4)
        nose_cur_loss_ = nose_cur_loss.item()
        jaw_cur_loss = criterion(outputs[5], y5)
        jaw_cur_loss_ = jaw_cur_loss.item()

        total_cur_loss = face_cur_loss + mouth_cur_loss + eyebrow_cur_loss + eye_cur_loss + nose_cur_loss + \
                         jaw_cur_loss
        total_cur_loss_ = total_cur_loss.item()

        face_loss.update(face_cur_loss_, sample_num)
        mouth_loss.update(mouth_cur_loss_, sample_num)
        eyebrow_loss.update(eyebrow_cur_loss_, sample_num)
        eye_loss.update(eye_cur_loss_, sample_num)
        nose_loss.update(nose_cur_loss_, sample_num)
        jaw_loss.update(jaw_cur_loss_, sample_num)
        total_loss.update(total_cur_loss_, sample_num)

        # Calculate correct
        correct_face_num = outputs[0].max(1)[1].eq(y0).sum().item()
        face_acc.update(correct_face_num, sample_num)
        correct_mouth_num = outputs[1].max(1)[1].eq(y1).sum().item()
        mouth_acc.update(correct_mouth_num, sample_num)
        correct_eyebrow_num = outputs[2].max(1)[1].eq(y2).sum().item()
        eyebrow_acc.update(correct_eyebrow_num, sample_num)
        correct_eye_num = outputs[3].max(1)[1].eq(y3).sum().item()
        eye_acc.update(correct_eye_num, sample_num)
        correct_nose_num = outputs[4].max(1)[1].eq(y4).sum().item()
        nose_acc.update(correct_nose_num, sample_num)
        correct_jaw_num = outputs[5].max(1)[1].eq(y5).sum().item()
        jaw_acc.update(correct_jaw_num, sample_num)

        total_acc.update(
            correct_face_num + correct_mouth_num + correct_eyebrow_num + correct_eye_num + correct_nose_num +
            correct_jaw_num, sample_num * 6)

        optimizer.zero_grad()
        total_cur_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=total_loss.avg),
        #                   acc=total_acc.avg, sample_num=sample_num)

        # plot progress
        bar.suffix = '(Epoch:{epoch} - {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.3f} | Acc: {acc:.3f}'.format(
            batch=batch_idx + 1,
            size=len(train_dataloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=total_loss.avg,
            acc=total_acc.avg,
            epoch=epoch
        )
        bar.next()
    bar.finish()

    return total_loss.avg, total_acc.avg


def validate(validate_dataloader, model, criterion, epoch, device):
    """Validating step"""
    model.eval()

    # Smooth loss function
    face_loss = AverageMeter()
    mouth_loss = AverageMeter()
    eyebrow_loss = AverageMeter()
    eye_loss = AverageMeter()
    nose_loss = AverageMeter()
    jaw_loss = AverageMeter()
    total_loss = AverageMeter()

    face_acc = AverageMeter()
    mouth_acc = AverageMeter()
    eyebrow_acc = AverageMeter()
    eye_acc = AverageMeter()
    nose_acc = AverageMeter()
    jaw_acc = AverageMeter()
    total_acc = AverageMeter()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    bar = Bar('Processing validate', max=len(validate_dataloader))
    with torch.no_grad():
        for batch_idx, (x, y0, y1, y2, y3, y4, y5) in enumerate(validate_dataloader):
            data_time.update(time.time() - end)

            x = x.to(device)
            # pytorch交叉熵损失函数内部自动设置one-hot编码格式
            y0 = y0.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            y4 = y4.to(device)
            y5 = y5.to(device)
            outputs = model(x)
            sample_num = x.size(0)

            face_cur_loss = criterion(outputs[0], y0)
            face_cur_loss_ = face_cur_loss.item()
            mouth_cur_loss = criterion(outputs[1], y1)
            mouth_cur_loss_ = mouth_cur_loss.item()
            eyebrow_cur_loss = criterion(outputs[2], y2)
            eyebrow_cur_loss_ = eyebrow_cur_loss.item()
            eye_cur_loss = criterion(outputs[3], y3)
            eye_cur_loss_ = eye_cur_loss.item()
            nose_cur_loss = criterion(outputs[4], y4)
            nose_cur_loss_ = nose_cur_loss.item()
            jaw_cur_loss = criterion(outputs[5], y5)
            jaw_cur_loss_ = jaw_cur_loss.item()

            total_cur_loss = face_cur_loss + mouth_cur_loss + eyebrow_cur_loss + eye_cur_loss + nose_cur_loss + \
                             jaw_cur_loss
            total_cur_loss_ = total_cur_loss.item()

            face_loss.update(face_cur_loss_, sample_num)
            mouth_loss.update(mouth_cur_loss_, sample_num)
            eyebrow_loss.update(eyebrow_cur_loss_, sample_num)
            eye_loss.update(eye_cur_loss_, sample_num)
            nose_loss.update(nose_cur_loss_, sample_num)
            jaw_loss.update(jaw_cur_loss_, sample_num)
            total_loss.update(total_cur_loss_, sample_num)

            # Calculate correct
            correct_face_num = outputs[0].max(1)[1].eq(y0).sum().item()
            face_acc.update(correct_face_num, sample_num)
            correct_mouth_num = outputs[1].max(1)[1].eq(y1).sum().item()
            mouth_acc.update(correct_mouth_num, sample_num)
            correct_eyebrow_num = outputs[2].max(1)[1].eq(y2).sum().item()
            eyebrow_acc.update(correct_eyebrow_num, sample_num)
            correct_eye_num = outputs[3].max(1)[1].eq(y3).sum().item()
            eye_acc.update(correct_eye_num, sample_num)
            correct_nose_num = outputs[4].max(1)[1].eq(y4).sum().item()
            nose_acc.update(correct_nose_num, sample_num)
            correct_jaw_num = outputs[5].max(1)[1].eq(y5).sum().item()
            jaw_acc.update(correct_jaw_num, sample_num)

            total_acc.update(
                correct_face_num + correct_mouth_num + correct_eyebrow_num + correct_eye_num + correct_nose_num +
                correct_jaw_num, sample_num * 6)

            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '(Epoch:{epoch} - {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.3f} | Acc: {acc:.3f}'.format(
                batch=batch_idx + 1,
                size=len(validate_dataloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=total_loss.avg,
                acc=total_acc.avg,
                epoch=epoch
            )
            bar.next()
        bar.finish()

    return total_loss.avg, total_acc.avg


def main():
    args = get_args()
    cfg.merge_from_file(args.yaml)
    cfg.freeze()
    start_epoch = 0
    # train_writer = None
    # val_writer = None
    best_val_acc = 0
    viz = Visdom()

    checkpoint_dir = Path(args.tensorboard)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print("[info] Creating model: {}".format(cfg.MODEL.ARCH))
    model = MultiTaskNet(model_name=cfg.MODEL.ARCH, num_embeddings=cfg.MODEL.EMBEDDINGS)

    if cfg.TRAIN.OPT.lower() == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check if the parameters can be trained
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # Retraining or training from a specific epoch
    resume_path = args.resume
    if resume_path:
        # if osp.isfile(resume_path):
        #     print("[info] loading checkpoint from '{}'".format(resume_path))
        #     checkpoint = torch.load(resume_path, map_location='cpu')
        #     start_epoch = checkpoint['epoch']
        #     model.load_state_dict(checkpoint['state_dict'])
        #     print("[info] Loaded checkpoint '{}' (epoch {})"
        #           .format(resume_path, checkpoint['epoch']))
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # else:
        #     print("[info] no checkpoint found at '{}'".format(resume_path))
        resume_checkpoint(model, resume_path)

    if device.type == 'cuda':
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    dataset = MultiTaskDataset(image_path=args.images_path, csv_path=args.labels_path,
                               img_size=cfg.MODEL.IMG_SIZE, transform=transforms.Compose(
            [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
             transforms.ToTensor(),
             transforms.Normalize(
                 # [0.485, 0.456, 0.406],
                 # [0.229, 0.224, 0.225])]))
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5])]))
    train_size = int(len(dataset) * (1.0 - args.ratio))
    validate_size = len(dataset) - train_size
    print("[info] Training data size: {} and validate data size: {}".format(train_size, validate_size))

    train_dataset, validate_dataset = random_split(dataset, [train_size, validate_size])
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.TRAIN.WORKERS)
    validate_dataloader = DataLoader(validate_dataset, batch_size=cfg.TEST.BATCH_SIZE, num_workers=cfg.TRAIN.WORKERS)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=0.1)

    if args.tensorboard is not None:
        # train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + cfg.MODEL.ARCH + "_train")
        # val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + cfg.MODEL.ARCH + "_val")
        viz.line([[0.0, 0.0]], [0.], win='train', opts=dict(title='Train loss && Acc.', legend=['loss', 'acc.']))
        viz.line([[0.0, 0.0]], [0.], win='validate', opts=dict(title='Validate loss && Acc.', legend=['loss', 'acc.']))

    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        # Train parameters setting
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, epoch, device)
        # Validate parameters setting
        val_loss, val_acc = validate(validate_dataloader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            # train_writer.add_scalar("loss", train_loss, epoch)
            # train_writer.add_scalar("acc", train_acc, epoch)
            # val_writer.add_scalar("loss", val_loss, epoch)
            # val_writer.add_scalar("acc", val_acc, epoch)
            viz.line([[train_loss, train_acc]], [epoch], win='train', update='append')
            viz.line([[val_loss, val_acc]], [epoch], win='validate', update='append')

            # for name, param in model.named_parameters():
            #     train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        if val_acc > best_val_acc:
            print(f"[epoch {epoch:02d}] best val acc was improved from {best_val_acc:.3f} to {val_acc:.3f}")
            model_state_dict = model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:02d}_{:.3f}_{:.3f}.pth".format(epoch, val_loss, val_acc)))
            )
            best_val_acc = val_acc
        else:
            print(f"[epoch {epoch:03d}] best val acc was not improved from {best_val_acc:.3f} ({val_acc:.3f})")

        scheduler.step(epoch)

    print("[info] Training finished")
    print(f"best val acc: {best_val_acc:.3f}")


if __name__ == '__main__':
    main()
