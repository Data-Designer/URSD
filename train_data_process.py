from os.path import join
import geopy
from geopy.distance import geodesic
from shapely import geometry
import pandas as pd
import argparse
import geopandas as gpd
import pandas as pd
import random
import json
import math
import requests
import time
import os
os.chdir('data')
random.seed(0)

with open('ozyx_ch_district_z15.json', 'r') as json_file_zyx:
    data_zyx = json.load(json_file_zyx)
    dat = pd.read_csv('odat_ch_district_z15.csv')
    asset_data = pd.DataFrame()
    road_data = pd.DataFrame()
    nightlight_data = pd.read_csv('nightlight/night_light.csv')
    asset_files = os.listdir('asset')
    for file in asset_files:
        if file[-3:] == 'csv':
            curr_area = file.split('_')[0]
            print('add ' + str(curr_area))

            belong_areaIDs = list(dat.loc[dat['major'] == curr_area, 'areaID'])
            area_data = pd.DataFrame()
            for curr_areaID in belong_areaIDs:
                curr_zyxs = data_zyx[str(curr_areaID)]
                curr_zyxs = [str(x[1])+'_'+str(x[2]) for x in curr_zyxs]
                curr_data = pd.DataFrame(curr_zyxs, columns=['image_id'])
                curr_data['area_id'] = curr_areaID
                area_data = pd.concat([area_data, curr_data], axis=0)

            cur_asset_data = pd.read_csv('asset/' + file)
            cur_asset_data = pd.merge(cur_asset_data, area_data, on='image_id', how='left')
            cur_asset_data['country'] = curr_area
            cur_road_data = pd.read_csv('road_network/' + file.split('_')[0] + '_road_network.csv')
            asset_data = pd.concat([asset_data, cur_asset_data], axis=0)
            road_data = pd.concat([road_data, cur_road_data], axis=0)

    asset_data = asset_data.drop_duplicates(['image_id'])
    road_data = road_data.drop_duplicates(['image_id'])
    nightlight_data.columns = ['image_id', 'light_sum', 'light_mean']
    data = pd.merge(asset_data, road_data, on='image_id')
    data = pd.merge(data, nightlight_data, on='image_id')
    data['exposure'] = data['exposure'].astype('int')
    data.to_csv('all_data_add_country.csv')
    has_exposure = data.loc[data['exposure'] != 0]
    has_no_exposure = data.loc[data['exposure'] == 0]
    data_select = pd.concat([has_exposure, has_no_exposure.sample(3000)], axis=0)
    data_select[['image_id', 'exposure', 'country', 'road_network', 'light_sum', 'light_mean']].to_csv('used_data_class5.csv', index=False)
