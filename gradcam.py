import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import warnings
warnings.filterwarnings("ignore")

from os import path, makedirs, listdir
os.environ['PYTHONHASHSEED'] = str(1)
import sys
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd
from tqdm import tqdm
import timeit

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from utils.utils import *
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils.zoo.new_model import *

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

model_name_pre = 'experiments/efficientnetb0'
snapshot_name_pre = 'classnum5_gradchange_noadd__test0.1_only_daytime_image_lr1e-05_wd0.01_gam0.9_dataall_batch32_dropput0.7_seed0'

model = Efficientnetmodel(5)

loaded_dict = torch.load(path.join(model_name_pre, snapshot_name_pre) + '/weight', map_location='cpu')['state_dict']
sd = model.state_dict()

for k in model.state_dict():
    if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        print(k)
        sd[k] = loaded_dict[k]
loaded_dict = sd
model.load_state_dict(loaded_dict)
best_score = torch.load(path.join(model_name_pre, snapshot_name_pre) + '/weight', map_location='cpu')['best_score']
print('load successfully')
print(best_score)

target_layer = [model.efficient._blocks[-1]]

data = pd.read_csv('data/all_data_inner.csv')
data['fn'] = data.apply(lambda x: x['country']+'/'+str(x['image_id'])+'.png', axis=1)

fn = list(data['fn'])

# select a picture
i = 1021
image_path = 'data/daytime_image/'+fn[i]
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
img = np.float32(img) / 255

# preprocess_image
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)

target_category = [ClassifierOutputTarget(0)]
grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0]
visualization = show_cam_on_image(img, grayscale_cam)
cv2.imwrite(f'image/satellite.jpg', visualization)
