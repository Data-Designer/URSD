import osm_export_tool
import osm_export_tool.tabular as tabular
from osm_export_tool.mapping import Mapping
# from osm_export_tool.geometry import load_geometry
# from osm_export_tool.sources import Overpass, Pbf, OsmExpress, OsmiumTool
# from osm_export_tool.package import create_package, create_posm_bundle
from os.path import join
import random
import geopy
from geopy.distance import geodesic
from shapely import geometry
import argparse
import geopandas as gpd
import pandas as pd
import json
import math
import requests
import time
import os
os.chdir('data')
random.seed(0)

# python asset.py --izyx ozyx_ch_district_z15 --idat odat_ch_district_z15 --odir asset

# input :
# input zyx_district.json, output dir
parser = argparse.ArgumentParser()
parser.add_argument('--izyx', help='input zyx_district json name')
parser.add_argument('--idat', help='input dat_district csv name')
parser.add_argument('--odir', help='output exposure dir name')

args = parser.parse_args()

if args.izyx == None or args.odir == None:
    print(
        "Error! please fill zyx_district json name, output exposure dir name")
    exit(0)

izyx = args.izyx
idat = args.idat
odir = args.odir


def load_and_process(file):
    d = gpd.read_file(file)
    return d


def get_different_building_type_area(data):
    data = data.to_crs(crs='EPSG:2380')
    # 住宅用房
    residential = (data['building'].apply(lambda x: x in ['apartments', 'residential', 'house', 'bungalow',
                                                          'detached', 'semidetached_house', 'dormitory']))
    residential_area = data.loc[residential].unary_union.area
    # 商业用房
    commercial = (data['shop'].notnull()) | (data['amenity'].apply(lambda x: x in ['marketplace',
                                                                                   'restaurant', 'fast_food', 'cafe',
                                                                                   'bar',
                                                                                   'pub'])) | (
                     data['building'].apply(lambda x: x in ['commercial', 'supermarket', 'retail', 'hotel']))
    commercial = commercial & (~residential)
    commercial_area = data.loc[commercial].unary_union.area
    # 办公用房
    office = (data['building'].apply(lambda x: x in ['office', 'school', 'university', 'hospital',
                                                     'college', 'government', 'public', 'civic',
                                                     'kindergarten'])) | (data['office'].notnull()) | (
                 data['amenity'].apply(lambda x: x in ['kindergarten',
                                                       'school', 'college', 'university', 'language_school', 'police',
                                                       'fire_station', 'bank',
                                                       'court_house', 'townhall', 'embassy', 'post_office']))
    office = office & (~residential) & (~commercial)
    office_area = data.loc[office].unary_union.area
    # 其他建筑类型
    others = (data['building'] != 'yes') & (~residential) & (~commercial) & (~office)
    other_area = data.loc[others].unary_union.area
    # 未标注
    unknown = (data['building'] == 'yes') & (~residential) & (~commercial) & (~office)
    unknown_area = data.loc[unknown].unary_union.area
    return [residential_area, commercial_area, office_area, other_area, unknown_area]


def get_exposure_data(data, building_type_value=[0.8544, 1.1150, 1.4385, 0.5351, 0.8737]):
    # building type are residential, commercial, office, other, unknown
    building_type_area = get_different_building_type_area(data.loc[data['building'].notnull()])
    return sum([building_type_area[i] * building_type_value[i] for i in range(len(building_type_area))])


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
    return gpd.GeoSeries([geometry.Polygon([point_left_up, point_right_up, point_right_down, point_left_down])], index=['geometry'])


def points2polygons(points, distance):
    image_id = []
    geometries = []
    left_bound = float('inf')
    down_bound = float('inf')
    right_bound = -float('inf')
    up_bound = -float('inf')
    for point in points:
        degree = num2deg(int(point.split('_')[1]), int(point.split('_')[0]), 15)
        if degree[1] < left_bound:
            left_bound = degree[1]
        if degree[1] > right_bound:
            right_bound = degree[1]
        if degree[0] < down_bound:
            down_bound = degree[0]
        if degree[0] > up_bound:
            up_bound = degree[0]
        image_id.append(point)
        geometries.append(get_polygon(degree[0], degree[1], distance))
    polygons = gpd.GeoDataFrame(geometries)
    polygons['image_id'] = image_id
    left_boundary = get_distance_point(down_bound, left_bound, distance, 270)[1]
    down_boundary = get_distance_point(down_bound, left_bound, distance, 180)[0]
    right_boundary = get_distance_point(up_bound, right_bound, distance, 90)[1]
    up_boundary = get_distance_point(up_bound, right_bound, distance, 0)[0]
    return polygons, [left_boundary, down_boundary, right_boundary, up_boundary]


def get_asset(curr_area, json_file, csv_file, select):
    with open(json_file, 'r') as json_file_zyx:
        data_zyx = json.load(json_file_zyx)
        da = pd.read_csv(csv_file)
        print('deal '+str(curr_area))

        belong_areaIDs = list(da.loc[da['major'] == curr_area, 'areaID'])
        total_zyxs = []
        for curr_areaID in belong_areaIDs:
            curr_zyxs = data_zyx[str(curr_areaID)]
            select_zyxs = random.sample(curr_zyxs, int(round(len(curr_zyxs) / select, 0)))
            total_zyxs.extend(select_zyxs)
        total = pd.DataFrame(total_zyxs, columns=['zoomlevel', 'image_id1', 'image_id2'])
        total['image_id'] = total.apply(lambda x: str(x[1])+'_'+str(x[2]), axis=1)
        total = total.drop_duplicates(['image_id'])
        total_zyxs = list(total['image_id'])

        geodata, box = points2polygons(total_zyxs, 1)
        geodata.crs = 'EPSG:4326'
        data = load_and_process('asset/' + str(curr_area) + '/asset_building_asset_polygon_polygons.shx')
        print('  start intersects')
        t = time.time()
        join_data = gpd.sjoin(data, geodata, op='intersects')
#         if join_data.shape[0]==0:
#             road_network=geodata[['image_id']]
#             road_network['road_network']=0
#             road_network['road_network']=road_network['road_network'].apply(lambda x:[0.0, 0.0, 0.0] if x==0 else x)
#             road_network.to_csv('./'+odir+'/'+str(curr_area)+'_road_network.csv', index=None)
#             return
        print('  end intersects and use '+str(time.time()-t)+'seconds')
        print('  start group')
        group = join_data.groupby(['image_id'])
        exposure = group.apply(get_exposure_data)
        exposure = pd.DataFrame(exposure)
        exposure['image_id'] = exposure.index
        exposure.columns = ['exposure', 'image_id']
        exposure.index = range(0, exposure.shape[0])
        exposure = pd.merge(geodata[['image_id']], exposure, on='image_id', how='left')
        exposure['exposure'].fillna(0, inplace=True)
        exposure.to_csv('./' + odir + '/' + str(curr_area) + '_exposure_data.csv', index=False)
        print(curr_area + ' exposure have been saved')


def get_asset_country_level(odir, country):
    exposures = pd.DataFrame()
    for curr_area in country['major_coun'].unique():
        belong_area = (country['major_coun'] == curr_area)
        data = load_and_process('./' + odir + '/' + str(curr_area) + '/asset_building_asset_polygon_polygons.shx')
        print('  start intersects')
        join_data = gpd.sjoin(data, country.loc[belong_area], op='intersects')
        print('  start group')
        group = join_data.groupby(['area_id'])
        exposure = group.apply(get_exposure_data)

        exposure = pd.DataFrame(exposure)
        exposure['area_id'] = exposure.index
        exposure.columns = ['exposure', 'area_id']
        exposure.index = range(0, exposure.shape[0])

        exposure = pd.merge(country.loc[belong_area], exposure, on='area_id', how='left')
        exposure['exposure'].fillna(0, inplace=True)
        exposures = pd.concat([exposures, exposure], axis=0)
    exposures.to_csv('./' + odir + '/' + 'country_level_exposure_data.csv', index=False)
    print(' exposure have been saved')


def main():
    t = time.time()
    # grid level asset
    data = pd.read_csv(idat + '.csv')
    for area in data['major'].unique():
        print(area)
        if area in ['Beijing Municipality', 'Guangdong Province',
            'Jiangsu Province', 'Shanghai Municipality',
            'Shandong Province', 'Tianjin Municipality', 'Zhejiang Province']:
            select = 10
        else:
            select = 50
        get_asset(area, izyx + '.json', idat + '.csv', select)
    print('total use ' + str(time.time() - t) + ' seconds')

    # country level asset
    country = gpd.read_file('country.shp')
    country = country.set_crs("EPSG:4326")
    country = country.to_crs(crs='EPSG:2380')
    country['area'] = country.area
    country = country.to_crs(crs='EPSG:4326')

    get_asset_country_level(odir, country)


if __name__ == '__main__':
    main()
