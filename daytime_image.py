import json
import csv
import os
os.chdir('data/')
import urllib.request
from multiprocessing import Process
import random
from time import sleep
import argparse
import pandas as pd

# cd lunwen/graduate/data/
# python daytime_image.py --token AAPK26c378f8297e41d28b9a71bee50ba8138O1_Eia2l0E7y0uzR2EJo64_r1wO29X_s4-ddBr5Qe1YDQ0P-GeKrOBL_xaUiE2D --odir daytime_image --zl 15

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--odir', help='output directory name')
parser.add_argument('--zl', help='zoom level')

args = parser.parse_args()

if args.token == None or args.odir == None:
    print(
        "Error! please fill all zyx_district json name, data_distric json name, temporal token information, output directory.")
    exit(0)

token = args.token
odir = args.odir
zl = 15
if args.zl is not None:
    zl = args.zl

base_url = "https://tiledbasemaps.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/"

if not os.path.isdir('./' + odir):
    os.mkdir('./' + odir)

# files = os.listdir('./' + 'asset')


def get_daytime_image(cur_area):
    print('deal ' + cur_area)
    image_id_list = list(pd.read_csv('./' + 'asset/' + cur_area + '_exposure_data.csv', index_col=None, header=0)['image_id'])
    print('total ' + str(len(image_id_list)) + ' district')
    if not os.path.isdir('./' + odir + '/' + cur_area):
        os.mkdir('./' + odir + '/' + cur_area)

    print('begin')
    for i in range(0, len(image_id_list)):
        url = base_url + str(zl) + '/' + image_id_list[i].split('_')[0] + '/' + image_id_list[i].split('_')[
            1] + '?token=' + token
        filename = './' + odir + '/' + cur_area + '/' + image_id_list[i] + '.png'
        # if except can not use
        if os.path.isfile(filename):
            continue
        try:
            urllib.request.urlretrieve(url, filename)
        except:
            sleep(1)
            urllib.request.urlretrieve(url, filename)
        print(str(image_id_list[i]) + ' has been download')
    print(cur_area + ' has been download')


data = pd.read_csv('odat_ch_district_z15.csv')
cur_areas = list(data['major'].unique())


if __name__ == "__main__":
    print('当前母进程: {}'.format(os.getpid()))
    p0 = Process(target=get_daytime_image, args=(cur_areas[0],))
    p1 = Process(target=get_daytime_image, args=(cur_areas[1],))
    p2 = Process(target=get_daytime_image, args=(cur_areas[2],))
    p3 = Process(target=get_daytime_image, args=(cur_areas[3],))
    p4 = Process(target=get_daytime_image, args=(cur_areas[4],))
    p5 = Process(target=get_daytime_image, args=(cur_areas[5],))
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()


