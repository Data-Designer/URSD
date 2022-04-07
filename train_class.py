import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
os.environ['PYTHONHASHSEED'] = str(1)

import sys
import numpy as np
import random
import torch
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import timeit
import cv2
import gc
import glob
# import torchvision.transforms as transforms

from utils.emailss import send_email
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from os import path, makedirs, listdir
# create new DataLoaderX class
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from apex import amp
from torch.nn import init
from utils.adamw import AdamW
from utils.losses import dice_round, ComboLoss
from tqdm import tqdm
from imgaug import augmenters as iaa
from utils.utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.utils import shuffle
from PIL import Image
# from torchvision import models
from sklearn import preprocessing
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
from utils.zoo.new_model import *

def seed_all():
    """seed everything"""
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True

seed_all()
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

model_name = 'experiments/efficientnetb0'
seed = 0
lr = 5e-04
wd = 0.01
gam = 0.9
dropout = 0.7
batch_size = 32
val_batch_size = 16
class_num = 5
snapshot_name = 'classnum{}_gradchange_noadd__test0.1_only_daytime_image_lr{}_wd{}_gam{}_data{}_batch{}_dropput{}_seed{}'.\
    format(class_num, lr, wd, gam, 'all', batch_size, dropout, seed)
makedirs(path.join(model_name, snapshot_name), exist_ok=True)
# default 'log_dir' is 'runs'
writer = SummaryWriter(path.join(model_name, snapshot_name) + "/runs")
# tensorboard --logdir=runs

load_pre = False
add_light = False
add_road = False


class TrainData(Dataset):
    """
    train dataset
    """
    def __init__(self, data_select, train_idxs, input_shape=(224, 224)):
        super().__init__()
        self.train_idxs = train_idxs
        self.elastic = iaa.ElasticTransformation(alpha=(0.25, 1.2), sigma=0.2)
        self.data = data_select
        self.input_shape = input_shape

    def __len__(self):
        return len(self.train_idxs)

    def __getitem__(self, idx):
        _idx = self.train_idxs[idx]
        _d = self.data.iloc[_idx, :]

        fn = 'data/daytime_image/' + _d[2] + '/' + _d[0] + '.png'
        if not os.path.exists(fn):
            print(fn)

        # read image
        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        # image enhancement
        if random.random() > 0.5:
            img = img[::-1, ...]

        if random.random() > 0.05:
            rot = random.randrange(4)
            if rot > 0:
                img = np.rot90(img, k=rot)

        if random.random() > 0.9:
            shift_pnt = (random.randint(-320, 320), random.randint(-320, 320))
            img = shift_image(img, shift_pnt)

        if random.random() > 0.9:
            rot_pnt = (img.shape[0] // 2 + random.randint(-320, 320), img.shape[1] // 2 + random.randint(-320, 320))
            scale = 0.9 + random.random() * 0.2
            angle = random.randint(0, 20) - 10
            if (angle != 0) or (scale != 1):
                img = rotate_image(img, angle, scale, rot_pnt)

        crop_size = img.shape[0]
        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.99:
            img = shift_channels(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.99:
            img = change_hsv(img, random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5))

        if random.random() > 0.99:
            if random.random() > 0.99:
                img = clahe(img)
            elif random.random() > 0.99:
                img = gauss_noise(img)
            elif random.random() > 0.99:
                img = cv2.blur(img, (3, 3))
        elif random.random() > 0.99:
            if random.random() > 0.99:
                img = saturation(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = brightness(img, 0.9 + random.random() * 0.2)
            elif random.random() > 0.99:
                img = contrast(img, 0.9 + random.random() * 0.2)

        if random.random() > 0.999:
            el_det = self.elastic.to_deterministic()
            img = el_det.augment_image(img)

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        sample = {'img': img,
                  'road': torch.tensor(np.log(np.array([eval(_d[3])]) + 1).reshape(-1, ), dtype=torch.float32),
                  'light': torch.tensor(np.log(np.array([_d[4], _d[5]]) + 1), dtype=torch.float32), 'exposure': _d[6],
                  'fn': fn}
        return sample


class ValData(Dataset):
    """val dataset"""
    def __init__(self,data_select, image_idxs, input_shape=(224, 224)):
        super().__init__()
        self.image_idxs = image_idxs
        self.data = data_select
        self.input_shape = input_shape

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]
        _d = self.data.iloc[_idx, :]

        fn = 'data/daytime_image/' + _d[2] + '/' + _d[0] + '.png'

        # read image
        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        crop_size = img.shape[0]
        if crop_size != self.input_shape[0]:
            img = cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        sample = {'img': img,
                  'road': torch.tensor(np.log(np.array([eval(_d[3])]) + 1).reshape(-1, ), dtype=torch.float32),
                  'light': torch.tensor(np.log(np.array([_d[4], _d[5]]) + 1), dtype=torch.float32), 'exposure': _d[6],
                  'fn': fn}
        return sample


def validate(net, val_loss, val_acc, val_acc_range1, data_loader, val_metric, loss_func, current_epoch):
    """val epoch """
    loss = []
    labels = []
    preds = []
    valid_acc = 0.0
    valid_acc_range1 = 0.0

    with torch.no_grad():
        for i, sample in enumerate(tqdm(data_loader)):
            label = sample["exposure"].cuda(non_blocking=True)
            labels.extend(label.cpu())
            img = sample["img"].cuda(non_blocking=True)
            nightlight = sample['light'].cuda(non_blocking=True)
            road = sample['road'].cuda(non_blocking=True)

            if add_light:
                if add_road:
                    out = net(img, nightlight, road)
                else:
                    out = net(img, nightlight)
            else:
                if add_road:
                    out = net(img, road)
                else:
                    out = net(img)

            loss.append(loss_func(out, label).cpu())

            # Compute the accuracy
            ret, prediction = torch.max(out, 1)
            preds.extend(prediction.cpu())
            correct_counts = prediction.eq(label.view_as(prediction))
            correct_counts_range1 = prediction.eq(label.view_as(prediction)) | prediction.eq(
                label.view_as(prediction) - 1) | prediction.eq(label.view_as(prediction) + 1)

            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            acc_range1 = torch.mean(correct_counts_range1.type(torch.FloatTensor))

            # Compute total accuracy in the whole batch and add to train_acc
            valid_acc += acc.item() * img.size(0)
            valid_acc_range1 += acc_range1.item() * img.size(0)

            # Compute other

    d = np.mean(loss)
    mses = mean_absolute_error(preds, labels)

    print('\n')
    print("nll loss: {}; acc: {}; acc +-1: {}; mse: {}".
          format(d, valid_acc/val_size, valid_acc_range1/val_size, mses))
    val_loss.append(d)
    val_acc.append(valid_acc/val_size)
    val_acc_range1.append(valid_acc_range1/val_size)
    writer.add_scalar("Test/loss", d, current_epoch)
    writer.add_scalar("Test/acc", valid_acc/val_size, current_epoch)
    writer.add_scalar("Test/acc +-1", valid_acc_range1 / val_size, current_epoch)
    return valid_acc/val_size, valid_acc_range1 / val_size, mses


def evaluate_val(data_val, best_score, model, val_metric, loss_func, snapshot_name, current_epoch, val_loss, val_acc, val_acc_range1):
    model = model.eval()
    d, acc_range1, mses = validate(model, val_loss, val_acc, val_acc_range1, data_loader=data_val, val_metric=val_metric, loss_func=loss_func, current_epoch=current_epoch)

    if d > best_score:
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'best_score': d,
            'acc_+-1': acc_range1,
            'mse': mses,
        }, path.join(path.join(model_name, snapshot_name) + '/weight'))
        best_score = d

    print("score: {}\tscore_best: {}".format(d, best_score))
    return best_score


def train_epoch(current_epoch, loss_func, model, optimizer, scheduler, train_data_loader, train_loss):
    losses = AverageMeter()

    iterator = tqdm(train_data_loader)
    model.train()
    for i, sample in enumerate(iterator):
        imgs = sample["img"].cuda(non_blocking=True)
        labels = sample["exposure"].cuda(non_blocking=True)
        nightlight = sample['light'].cuda(non_blocking=True)
        road = sample['road'].cuda(non_blocking=True)

        if add_light:
            if add_road:
                out = model(imgs, nightlight, road)
            else:
                out = model(imgs, nightlight)
        else:
            if add_road:
                out = model(imgs, road)
            else:
                out = model(imgs)

        loss = loss_func(out, labels)

        losses.update(loss.item(), imgs.size(0))

        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss {loss.val:.4f} {loss.avg:.4f})".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses))

        optimizer.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.1)
        optimizer.step()

    scheduler.step(current_epoch)
    train_loss.append(losses.avg)

    print('\n')
    print("epoch: {}; lr {:.7f}; Loss {loss.avg:.4f}".format(
        current_epoch, scheduler.get_lr()[-1], loss=losses))
    writer.add_scalar("Training/loss", losses.avg, current_epoch)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    input_shape = (224, 224)
    data = pd.read_csv('data/used_data.csv', encoding='utf-8')
    has_exposure = data.loc[data['exposure'] != 0]
    has_no_exposure = data.loc[data['exposure'] == 0]

    val_min = np.percentile(has_exposure["exposure"], 0)
    val_per25 = np.percentile(has_exposure["exposure"], 25)
    val_per50 = np.percentile(has_exposure["exposure"], 50)
    val_per75 = np.percentile(has_exposure["exposure"], 75)
    val_max = np.percentile(has_exposure["exposure"], 100)
    has_exposure['exposure_class'] = np.digitize(has_exposure['exposure'],
                                                 [val_min, val_per25, val_per50, val_per75])
    has_no_exposure['exposure_class'] = 0

    data_select = pd.concat([has_exposure, has_no_exposure], axis=0)
    data_select = shuffle(data_select)
    models_folder = 'experiments'
    makedirs(models_folder, exist_ok=True)
    makedirs(model_name, exist_ok=True)

    train_loss = []
    val_loss = []
    val_acc = []
    val_acc_range1 = []

    t0 = timeit.default_timer()
    train_idxs, val_idxs = train_test_split(np.arange(data_select.shape[0]), test_size=0.1, random_state=seed)


    train_size = len(train_idxs)
    val_size = len(val_idxs)
    steps_per_epoch = len(train_idxs) // batch_size
    validation_steps = len(val_idxs) // val_batch_size

    print('steps_per_epoch', steps_per_epoch, 'validation_steps', validation_steps)

    data_train = TrainData(data_select, train_idxs)
    data_val = ValData(data_select, val_idxs)

    train_data_loader = DataLoaderX(data_train, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=False)
    val_data_loader = DataLoaderX(data_val, batch_size=val_batch_size, num_workers=0, shuffle=False, pin_memory=False)

    if add_light:
        if add_road:
            model = Efficientnetmodel(class_num, nightlight=True, road=True)
        else:
            model = Efficientnetmodel(class_num, nightlight=True)
    else:
        if add_road:
            model = Efficientnetmodel(class_num, road=True)
        else:
            model = Efficientnetmodel(class_num)

    if load_pre:
        model_name_pre = 'experiments/efficientnetb0'
        snapshot_name_pre = 'classnum5_gradchange_noadd__test0.1_only_daytime_image_lr1e-05_wd0.01_gam0.9_dataall_batch32_dropput0.7_seed0'

        loaded_dict = torch.load(path.join(model_name_pre, snapshot_name_pre) + '/weight')['state_dict']
        sd = model.state_dict()

        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                print(k)
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        best_score = torch.load(path.join(model_name_pre, snapshot_name_pre) + '/weight')['best_score']
        print('load successfully')

        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
                print(name + 'no grad')

        for name, param in model.named_parameters():
            if 'fc' in name and 'weight' in name:
                print(name + 'has grad')
                init.kaiming_uniform_(param, a=0, mode='fan_in', nonlinearity='leaky_relu')
    else:
        best_score = 0

    model = model.cuda()

    params = model.parameters()

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 140, 160, 180], gamma=gam)

    nll_loss = nn.CrossEntropyLoss().cuda()

    val_metric = ['mean_loss', 'r2']

    _cnt = -1

    torch.cuda.empty_cache()
    for epoch in range(150):
        train_epoch(epoch, nll_loss, model, optimizer, scheduler, train_data_loader, train_loss)
        if epoch % 1 == 0:
            _cnt += 1
            torch.cuda.empty_cache()
            best_score = evaluate_val(val_data_loader, best_score, model, val_metric, nll_loss, snapshot_name, epoch, val_loss, val_acc, val_acc_range1)
    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))

    out = pd.DataFrame([train_loss, val_loss, val_acc, val_acc_range1]).T
    out.columns = ['train_loss', 'val_loss', 'val_acc', 'val_acc+-1']
    out.to_csv(path.join(model_name, snapshot_name) + '/result.csv', index=None)

    send_email(list(range(150)), train_loss, val_loss, val_acc, val_acc_range1)  # 发送邮件





