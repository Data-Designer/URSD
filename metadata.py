# -*- coding:utf-8 -*-
import json
import urllib.request
import os
os.chdir('data')
from os import makedirs
import math
import requests
import argparse
import collections
import pandas as pd
from time import sleep

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely import geometry
import geopandas as gpd

# python metadata.py --zl 15 --ozyx ozyx_ch_district_z15 --odat odat_ch_district_z15 --token AAPK26c378f8297e41d28b9a71bee50ba8138O1_Eia2l0E7y0uzR2EJo64_r1wO29X_s4-ddBr5Qe1YDQ0P-GeKrOBL_xaUiE2D

# input :
# zoom level, output zyx_district.json, data_district.json
parser = argparse.ArgumentParser()
parser.add_argument('--zl', help='zoom level')
parser.add_argument('--ozyx', help='output zyx_district json name')
parser.add_argument('--odat', help='output data_district csv name')
parser.add_argument('--token', help='input temporal token')
parser.add_argument('--nation', help='enter target nation code. Default = KR')
parser.add_argument('--minor', help='minor county, Default = KR.Districts')
parser.add_argument('--dset', help='enter target dataset Default = KOR_MBR_2018')
parser.add_argument('--dcoll', help='Inner dataCollcetion name, Default = Gender')
parser.add_argument('--start', help='for separable credit use, input start if you want')
parser.add_argument('--end', help='for separable credit use, input end if you want')

args = parser.parse_args()

if args.zl == None or args.ozyx == None or args.odat == None or args.token == None:
    print(
        "Error! please fill all zoom level, zyx_district json name, data_distric json name, temporal token information")
    exit(0)

zl = int(args.zl)
ozyx = args.ozyx
odat = args.odat
token = args.token
nation = 'CHN'
minor = 'CN.CountyLevel'
dset = 'CHN_MBR_2020'
dcoll = 'Gender'
geographyids = str(['01'])

if args.nation is not None:
    nation = args.nation
if args.minor is not None:
    minor = args.minor
if args.dset is not None:
    dset = args.dset
if args.dcoll is not None:
    dcoll = args.dcoll
dcoll = str([dcoll])

start = -1
end = -1
start_bool = False
end_bool = False

if args.start is not None:
    start = int(args.start)
    start_bool = True
    if start < 0:
        print("You put wrong start; put non-negative number")
        exit(0)
if args.end is not None:
    end = int(args.end)
    end_bool = True
    if end < 0:
        print("You put wrong end; put non-negative number that is larger than start")
        exit(0)


def extractZYX(zoomlevel, polygon):
    xlist = []
    ylist = []
    result = []
    for lnglat in polygon:
        # print(lnglat)
        lat = lnglat[1]
        lng = lnglat[0]
        xtile, ytile = deg2num(lat, lng, zoomlevel)
        # print(xtile,ytile)
        xlist.append(xtile)
        ylist.append(ytile)

    xmin, xmax, ymin, ymax = min(xlist), max(xlist), min(ylist), max(ylist)
    new_xlist = list(range(xmin, xmax))
    new_ylist = list(range(ymin, ymax))

    print(xmin, xmax, ymin, ymax)
    tf_table = [tf[:] for tf in [[None] * (xmax - xmin + 1)] * (ymax - ymin + 1)]

    for y in list(range(ymin, ymax + 1)):
        for x in list(range(xmin, xmax + 1)):
            lat, lng = num2deg(x, y, zoomlevel)
            tf_table[y - ymin][x - xmin] = 1 if Polygon(polygon).contains(Point(lng, lat)) else 0

    # return z,y,x only when more than 3 corners of image are inside the district boundary
    for y in new_ylist:
        for x in new_xlist:
            if tf_table[y - ymin][x - xmin] + tf_table[y - ymin][x - xmin + 1] + tf_table[y - ymin + 1][x - xmin] + \
                    tf_table[y - ymin + 1][x - xmin + 1] >= 3:
                result.append([zoomlevel, y, x])
    # print(tf_table)
    return result


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
# print(kr_translation.eng2kor_district)
# Use 9-alpha on slack

url = 'https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/StandardGeographyQuery/'
params = {'f': 'pjson',
          'sourceCountry': nation,
          'token': token,
          # 'geographyLayers':major,
          'geographyids': geographyids,
          'returnSubGeographyLayer': 'true',
          'subGeographyLayer': minor,
          # 'DatasetID':dset,
          'returnGeometry': 'true',
          'optionalCountryHierarchy': 'census'
          }
print(params)

res = requests.get(url, params=params)
res_json = res.json()
# print(res.text)
areaID_features = res_json['results'][0]['value']['features']

# get shp file
area_id = []
country_name = []
major_country_name = []
geometries = []
for i in range(len(areaID_features)):
    country_name.append(areaID_features[i]['attributes']['AreaName'])
    area_id.append(areaID_features[i]['attributes']['AreaID'])
    major_country_name.append(areaID_features[i]['attributes']['MajorSubdivisionName'])
    geometries.append(gpd.GeoSeries([geometry.Polygon(areaID_features[i]['geometry']['rings'][0])], index=['geometry']))
polygons = gpd.GeoDataFrame(geometries)
polygons['country_na'] = country_name
polygons['area_id'] = area_id
polygons['major_coun'] = major_country_name
polygons.to_file(driver='ESRI Shapefile', filename=r'country.shp')


# get grid
ozyx_dict = {}
area_dict = {}

# print(areaID_features[0])
for af in areaID_features:
    # if the area comes from different data collection, continue
    if af['attributes']['DatasetID'] != dset:
        continue
    # geometry
    zyx_list = extractZYX(zl, af['geometry']['rings'][0])
    print(af['attributes']['AreaName'], len(zyx_list))
    ozyx_dict[af['attributes']['AreaID']] = zyx_list
    area_dict[af['attributes']['AreaID']] = {'minor': af['attributes']['AreaName'],
                                             'major': af['attributes']['MajorSubdivisionName']}

ordered_ozyx_dict = collections.OrderedDict(sorted(ozyx_dict.items()))
ordered_areaID_list = list(ordered_ozyx_dict.keys())

print('number of minors')
print(len(ordered_areaID_list))

with open(ozyx + '.json', 'w') as json_file:
    json.dump(ordered_ozyx_dict, json_file)

if start_bool == False:
    start = 0
if end_bool == False:
    end = len(ordered_areaID_list)

selected_areaID_list = ordered_areaID_list[start:end]
print('selected_list')
print(selected_areaID_list)
print(len(selected_areaID_list))

sleep(5)

# get base information
total_times = int(len(selected_areaID_list) / 500) + 1
df = pd.DataFrame()
for i in range(total_times):
    selected_areaID_list_part = selected_areaID_list[i * 500: (i + 1) * 500]
    url = 'https://geoenrich.arcgis.com/arcgis/rest/services/World/geoenrichmentserver/Geoenrichment/Enrich?'
    url += 'f=pjson&'
    url += 'studyAreas=[{'
    url += '"sourceCountry":"{}","layer":"{}","ids":{}'.format(nation, minor, str(selected_areaID_list_part))
    url += '}]&'
    url += 'dataCollections={}&'.format(dcoll)
    url += 'inSR=4326&outSR=4326&'
    url += 'token={}'.format(token)

    # print(url)
    res = requests.get(url)
    # print(res.text)
    res_json = res.json()

    fields = res_json['results'][0]['value']['FeatureSet'][0]['fields']

    code_list = []
    for attribute_dict in fields:
        try:
            # only attributes have fullname
            fullname = attribute_dict["fullName"]
            code_list.append(attribute_dict["name"])
            # desc_list.append(attribute_dict["alias"])
        except:
            continue

    print(code_list)

    pandas_baseline = {'areaID': []}
    for code in code_list:
        pandas_baseline[code] = []

    attributes_list = res_json['results'][0]['value']['FeatureSet'][0]['features']

    for attr in attributes_list:
        attributes = attr['attributes']
        pandas_baseline['areaID'].append(str(attributes['StdGeographyID']))
        for code in code_list:
            pandas_baseline[code].append(attributes[code])

    # print(pandas_baseline)

    baseline_minor = []
    baseline_major = []
    for areaID in pandas_baseline['areaID']:
        baseline_minor.append(area_dict[areaID]['minor'])
        baseline_major.append(area_dict[areaID]['major'])
    pandas_baseline['minor'] = baseline_minor
    pandas_baseline['major'] = baseline_major

    print(pandas_baseline)

    # pandas work test

    df = pd.concat([df, pd.DataFrame.from_dict(pandas_baseline)], axis=0)
s_name = ""
e_name = ""

if start_bool == True:
    s_name = "_start_" + str(start)
if end_bool == True:
    e_name = "_end_" + str(end)
df.to_csv(odat + s_name + e_name + '.csv', index=False)

makedirs('asset/', exist_ok=True)
makedirs('road_network/', exist_ok=True)
for area in df['major'].unique():
    makedirs('asset/' + area, exist_ok=True)
    makedirs('road_network/' + area, exist_ok=True)
