import json
import os
os.chdir('data')
import glob
import argparse
from PIL import Image
import requests
import urllib.request
import math

# cd /home/xuyunhui/lunwen/graduate/data
# python night_light.py --token AAPK26c378f8297e41d28b9a71bee50ba8138O1_Eia2l0E7y0uzR2EJo64_r1wO29X_s4-ddBr5Qe1YDQ0P-GeKrOBL_xaUiE2D --ddir daytime_image --ondir night_light --cndir night_light_crop --zl 9 --zdiff 6 --nation CHN

parser = argparse.ArgumentParser()
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--ddir', help='input daytime directory name')
parser.add_argument('--ondir', help='output directory name')
parser.add_argument('--cndir', help='output directory changed name')
parser.add_argument('--zl', help='zoom level')
parser.add_argument('--zdiff', help='zoom level diff')
parser.add_argument('--nation', help='nation')

args = parser.parse_args()

token = args.token
ddir = args.ddir
ondir = args.ondir
cndir = args.cndir
nation = args.nation
zl = 9
zdiff= 6
if args.zl is not None:
    zl = int(args.zl)
if args.zdiff is not None:
    zdiff = int(args.zdiff)

if not os.path.isdir('./'+ondir):
    os.mkdir('./'+ondir)


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


ddir_list = glob.glob(ddir+'/*/*.png')
print('load success')

if not os.path.isdir(cndir):
    os.mkdir(cndir)

for ddir_image in ddir_list:
    sp = ddir_image.split('/')
    #sp = ddir_image.split('\\')
    curr_img = sp[-1]
    curr_ddir = sp[-2]
    curr_out_path = cndir+'/'+curr_ddir
    if not os.path.isdir(curr_out_path):
        os.mkdir(curr_out_path)
    curr_y = int(curr_img.split('_')[0])
    curr_x = int(curr_img.split('_')[1].split('.')[0])
    night_y = curr_y//2**zdiff
    night_x = curr_x//2**zdiff
    pixel_y = curr_y%2**zdiff
    pixel_x = curr_x%2**zdiff

    try:
        curr_nightimg = Image.open('./'+ondir+'/'+str(night_y)+'_'+str(night_x)+'.png')
    except:
        urllib.request.urlretrieve('https://tiles.arcgis.com/tiles/P3ePLMYs2RVChkJx/arcgis/rest/services/Earth_at_Night_2016/MapServer/tile/{}/{}/{}'.format(str(zl),str(night_y),str(night_x)),'./'+ondir+'/{}_{}.png'.format(str(night_y),str(night_x)))
        curr_nightimg = Image.open('./'+ondir+'/'+str(night_y)+'_'+str(night_x)+'.png')
    unit = curr_nightimg.size[0]//2**zdiff
    curr_area = (pixel_x*unit, pixel_y*unit, pixel_x*unit+unit, pixel_y*unit+unit)
    curr_cropped_nightimg = curr_nightimg.crop(curr_area)
    curr_cropped_nightimg.save(curr_out_path+'/'+curr_img)
    print(curr_ddir, curr_img, "converted to "+curr_out_path+'/'+curr_img)

