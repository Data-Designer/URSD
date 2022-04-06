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

izyx = 'ozyx_ch_district_z15'
idat = 'odat_ch_district_z15'
odir = 'road_network'


def load_and_process(file):
    d = gpd.read_file(file)
    return d


def get_road_length(data):
    data = data.to_crs(crs='EPSG:2380')
    road_length = []
    road_length.append(data.loc[data['highway'].notnull()].unary_union.length)
    road_length.append(data.loc[data['railway'].notnull()].unary_union.length)
    road_length.append(data.loc[data['bridge'].notnull()].unary_union.length)
    return road_length


def get_distance_point(lat, lon, distance, direction):
    """
    根据经纬度，距离，方向获得一个地点
    :param lat: 纬度
    :param lon: 经度
    :param distance: 距离（千米）
    :param direction: 方向（北：0，东：90，南：180，西：270）
    :return:
    """
    start = geopy.Point(lat, lon)
    d = geodesic(kilometers=distance)
    return d.destination(point=start, bearing=direction)


def get_round_box(lat, lon, distance):
    left = get_distance_point(lat, lon, distance, 270)[1]
    up = get_distance_point(lat, lon, distance, 0)[0]
    right = get_distance_point(lat, lon, distance, 90)[1]
    down = get_distance_point(lat, lon, distance, 180)[0]
    return left, down, right, up


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)


def get_polygon(lat, lon, distance):
    left = get_distance_point(lat, lon, distance, 270)[1]
    up = get_distance_point(lat, lon, distance, 0)[0]
    right = get_distance_point(lat, lon, distance, 90)[1]
    down = get_distance_point(lat, lon, distance, 180)[0]
    point_left_up = (left, up)
    point_right_up = (right, up)
    point_left_down = (left, down)
    point_right_down = (right, down)
    return gpd.GeoSeries([geometry.Polygon([point_left_up, point_right_up, point_right_down, point_left_down])],
                         index=['geometry'])


def points2polygons(points, distance=0.6):
    image_id = []
    geometries = []
    left_bound = float('inf')
    down_bound = float('inf')
    right_bound = -float('inf')
    up_bound = -float('inf')
    for point in points:
        degree = num2deg(point[1], point[0], 15)
        if degree[1] < left_bound:
            left_bound = degree[1]
        if degree[1] > right_bound:
            right_bound = degree[1]
        if degree[0] < down_bound:
            down_bound = degree[0]
        if degree[0] > up_bound:
            up_bound = degree[0]
        image_id.append(str(point[0]) + '_' + str(point[1]))
        geometries.append(get_polygon(degree[0], degree[1], distance))
    polygons = gpd.GeoDataFrame(geometries)
    polygons['image_id'] = image_id
    left_boundary = get_distance_point(down_bound, left_bound, distance, 270)[1]
    down_boundary = get_distance_point(down_bound, left_bound, distance, 180)[0]
    right_boundary = get_distance_point(up_bound, right_bound, distance, 90)[1]
    up_boundary = get_distance_point(up_bound, right_bound, distance, 0)[0]
    return polygons, [left_boundary, down_boundary, right_boundary, up_boundary]


def get_road_network(odir, curr_area):
    dat = pd.read_csv('asset/' + curr_area + '_exposure_data.csv')
    total_zyxs = list(dat['image_id'].apply(lambda x: [int(x.split('_')[0]), int(x.split('_')[1])]))

    #     if os.path.isfile('./'+odir+'/'+str(curr_area)+'_road_network.csv'):
    #         print('has saved')
    #         return

    geodata, box = points2polygons(total_zyxs, 0.6)
    geodata.crs = 'EPSG:4326'
    data = load_and_process('./' + odir + '/' + str(curr_area) + '/road_network_planet_osm_line_lines.shx')
    print('  start intersects')
    t = time.time()
    join_data = gpd.sjoin(data, geodata, op='intersects')
    if join_data.shape[0] == 0:
        road_network = geodata[['image_id']]
        road_network['road_network'] = 0
        road_network['road_network'] = road_network['road_network'].apply(lambda x: [0.0, 0.0, 0.0] if x == 0 else x)
        road_network.to_csv('./' + odir + '/' + str(curr_area) + '_road_network.csv', index=None)
        return
    print('  end intersects and use ' + str(time.time() - t) + 'seconds')
    print('  start group')
    group = join_data.groupby(['image_id'])
    road_network = group.apply(get_road_length)
    road_network = pd.DataFrame(road_network)
    road_network['image_id'] = road_network.index
    road_network.columns = ['road_network', 'image_id']
    road_network.index = range(0, road_network.shape[0])
    road_network = pd.merge(geodata[['image_id']], road_network, on='image_id', how='left')
    road_network['road_network'].fillna(0, inplace=True)
    road_network['road_network'] = road_network['road_network'].apply(lambda x: [0.0, 0.0, 0.0] if x == 0 else x)
    road_network.to_csv('./' + odir + '/' + str(curr_area) + '_road_network.csv', index=None)
    print(' road_network have been saved')


data = pd.read_csv(idat + '.csv')
for area in data['major'].unique():
    get_road_network('road_network', area)
