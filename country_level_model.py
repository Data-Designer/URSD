import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

os.environ['PYTHONHASHSEED'] = str(1)
import numpy as np
np.random.seed(1)
import random
random.seed(1)

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from utils.zoo.new_model import *
from utils.utils import *

from os import path, makedirs, listdir
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import pandas as pd
from tqdm import tqdm
import cv2


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

input_shape = (224, 224)
data = pd.read_csv('data/all_data_add_country.csv')
data = data.iloc[:, 1:]

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
data_all = pd.concat([has_exposure, has_no_exposure], axis=0)
data_all.reset_index(inplace=True)
data_all = data_all.iloc[:, 1:]

data_root_path = 'data/'
model_name = 'experiments/add_feature_efficientnetb0'
snapshot_name = 'classnum5_gradchange_noadd__test0.1_only_daytime_image_lr0.0001_wd0.01_gam0.9_dataall_batch64_dropput0.7_seed0'

add_light = True
add_road = True
class_num = 5


class ValData(Dataset):
    def __init__(self, data, image_idxs):
        super().__init__()
        self.image_idxs = image_idxs
        self.data = data

    def __len__(self):
        return len(self.image_idxs)

    def __getitem__(self, idx):
        _idx = self.image_idxs[idx]
        _d = self.data.iloc[_idx, :]

        fn = data_root_path + 'daytime_image/' + _d[3] + '/' + _d[0] + '.png'

        # 读入图像
        img = cv2.imread(fn, cv2.IMREAD_COLOR)

        crop_size = img.shape[0]
        if crop_size != input_shape[0]:
            img = cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR)

        img = preprocess_inputs(img)

        img = torch.from_numpy(img.transpose((2, 0, 1))).float()

        sample = {'img': img,
                  'road': torch.tensor(np.log(np.array([eval(_d[4])]) + 1).reshape(-1, ), dtype=torch.float32),
                  'light': torch.tensor(np.log(np.array([_d[5], _d[6]]) + 1), dtype=torch.float32), 'exposure': _d[7],
                  'fn': fn}
        return sample

# get inner information
batch_size = 16
data_loader = DataLoader(ValData(data_all, np.arange(data_all.shape[0])), batch_size=batch_size, num_workers=0, shuffle=False, pin_memory=False)
torch.cuda.empty_cache()

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
model = model.cuda()

loaded_dict = torch.load(path.join(model_name, snapshot_name) + '/weight')['state_dict']
sd = model.state_dict()

for k in model.state_dict():
    if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
        print(k)
        sd[k] = loaded_dict[k]
loaded_dict = sd
model.load_state_dict(loaded_dict)
best_score = torch.load(path.join(model_name, snapshot_name) + '/weight')['best_score']
print('load successfully')

iterator = tqdm(data_loader)
model.eval()
labels = []
inners = []
preds = []
ids = []
for i, sample in enumerate(iterator):
    label = sample["exposure"].cuda(non_blocking=True)
    labels.extend(label.cpu())
    ids.extend(sample["fn"])
    img = sample["img"].cuda(non_blocking=True)
    nightlight = sample['light'].cuda(non_blocking=True)
    road = sample['road'].cuda(non_blocking=True)

    if add_light:
        if add_road:
            out = model(img, nightlight, road)
        else:
            out = model(img, nightlight)
    else:
        if add_road:
            out = model(img, road)
        else:
            out = model(img)

    inners.extend(out.cpu().detach().numpy())
    ret, prediction = torch.max(out, 1)
    preds.extend(prediction.cpu())

dd = pd.DataFrame([ids, preds, labels, inners]).T
dd.columns = ['ids', 'preds', 'labels', 'inners']
dd['image_id'] = dd['ids'].apply(lambda x: x.split('/')[-1].strip('.png'))
dd['pred'] = dd['preds'].apply(lambda x: x.item())
dd['label'] = dd['labels'].apply(lambda x: x.item())
dd['i1'] = dd['inners'].apply(lambda x: x[0])
dd['i2'] = dd['inners'].apply(lambda x: x[1])
dd['i3'] = dd['inners'].apply(lambda x: x[2])
dd['i4'] = dd['inners'].apply(lambda x: x[3])
dd['i5'] = dd['inners'].apply(lambda x: x[4])
dd.drop(['ids', 'preds', 'labels', 'inners'], axis=1, inplace=True)

data_out = pd.merge(dd, data_all, on='image_id')
data_out.to_csv('data/all_data_inner.csv', index=False)  # save inner information

# data_process and build model
data = data_out.copy()
country_inner = data.groupby('area_id').agg({'i1': ['count', 'mean'], 'i2': ['mean'], 'i3': ['mean'],
                                            'i4': ['mean'], 'i5': ['mean']})
country_inner = pd.DataFrame(country_inner)
country_inner['area_id'] = country_inner.index
country_inner.columns = ['count_num', 'i1', 'i2', 'i3', 'i4', 'i5', 'area_id']
country_inner.index = range(0, country_inner.shape[0])
country_exposure = pd.read_csv('data/asset/country_level_exposure_data.csv')
country_exposure['exposure_per'] = country_exposure['exposure']/country_exposure['area']
country_inter = pd.merge(country_exposure, country_inner, on='area_id')

scaler = MinMaxScaler()
scaler.fit(country_inter[['exposure_per']])
country_inter[['exposure_per']] = scaler.transform(country_inter[['exposure_per']])
xtrain, xtest, ytrain, ytest = train_test_split(country_inter[['i1',
                                                               'i2', 'i3', 'i4', 'i5', 'count_num']],
                                                country_inter['exposure_per'], train_size=0.70, random_state=0)
model = XGBR(max_depth=5, learning_rate=0.01, n_estimators=171, objective='reg:gamma', min_child_weight=4, subsample=1,
             colsample_bytree=1, gamma=0.9, reg_alpha=0.1, reg_lambda=1)
model.fit(xtrain, ytrain)

train_preds = model.predict(xtrain)
test_preds = model.predict(xtest)

preds = model.predict(country_inter[['i1', 'i2', 'i3', 'i4', 'i5', 'count_num']])
print(mean_squared_error(ytrain, train_preds))
print(r2_score(ytrain, train_preds))
print(mean_squared_error(ytest, test_preds))
print(r2_score(ytest, test_preds))
