import argparse
import os
from tqdm import tqdm
from imantics import Mask, Image, Category, Dataset
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
import os, os.path
import numpy as np
import cv2
from utils.utils_path import is_pathname_valid
import pathlib
import json
from itertools import zip_longest
import argparse


class Lepu2COCO(object):
    def __init__(self, excel_path, save_path):
        self.data_frame = pd.read_excel(excel_path)
        self.save_path = save_path
        self.category_mapping = {'left_ventricular': 1, 'left_atrium': 2}
        self.left_ventricular = ['1.5.1', '1.5.2', '1.5.3', '1.5.4', '1.5.5', '1.5.6', '1.5.7', '1.5.8']  # 左心室
        self.left_atrium = ['5.1.1', '5.1.2', '5.1.3', '5.1.4', '5.1.4', '5.1.6', '5.1.7', '5.1.8']  # 左心房

    def _get_roi(self, ds):
        series = ds.pixel_array  # 像素值矩阵
        if not ds['PhotometricInterpretation'].value == 'RGB':  # 'RGB', 'YBR_FULL', 'YBR_FULL_422' TODO : checkpoints
            series = convert_color_space(series, ds['PhotometricInterpretation'].value, 'RGB', per_frame=True)
        if len(series.shape) == 2:
            series = np.expand_dims(series, axis=0)
        if len(series.shape) == 3:
            series = np.expand_dims(series, axis=series.shape[-1] != 3 and 3)
        ROI_info = ds.SequenceOfUltrasoundRegions[0]
        x_max, y_max, x_min, y_min = ROI_info.RegionLocationMaxX1, ROI_info.RegionLocationMaxY1, ROI_info.RegionLocationMinX0, ROI_info.RegionLocationMinY0
        series = series[:, y_min:y_max, x_min:x_max, :]
        frames, height, width, channels = series.shape
        if channels == 1:
            series = np.concatenate([cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB) for slice in series])
        # zero padding to get a square matrix, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0))
        how_to_padding = ((0, 0), (0, 0), (0, height - width), (0, 0)) if height > width else (
            (0, 0), (0, width - height), (0, 0), (0, 0))
        series = np.pad(series, how_to_padding, mode='constant', constant_values=0)
        return series

    def _get_polygons(self, label_content_json):
        polygons = {}
        label_content = json.loads(label_content_json)
        for region in label_content['regions']:
            if region['shape_attributes']['name'] == 'dotted' and region['region_attributes'][0][
                'type'] in self.left_ventricular:
                polygons['left_ventricular'] = list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
            elif region['shape_attributes']['name'] == 'dotted' and region['region_attributes'][0]['type'] in self.left_atrium:
                polygons['left_atrium'] = list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
        # assert len(mask_regions) == 2, 'wrong mask_regions length, should be 2, got %d' % len(mask_regions)

        return polygons

    def get_dataset(self, dataset_type='coco'):
        """
        Get dataset
        :param dataset:
        :return:
        """

        dataset = Dataset('lepu_A2C_A4C_seg')  # 先定义一个数据库对象，后续需要往里面添加具体的image和annotation
        pre_dicom_path = None
        for index, (frame_index, file_name, dicom_path, label_content_json) in enumerate(
                zip(self.data_frame['FrameIndex'],
                    self.data_frame['instanceID'],
                    self.data_frame['FilePath'],
                    self.data_frame['LabelContent'])):
            if not pre_dicom_path == dicom_path:
                ds = pydicom.read_file(dicom_path)  # DICOM文件的位置
                series = self._get_roi(ds)  # N,H,W,C( frames, height, width, channels)
                pre_dicom_path = dicom_path
            # image = series[frame_index]
            image = Image(series[frame_index - 1], id=index)
            image.file_name = '{}'.format(file_name)  # 为上面的Image对象添加coco标签格式的'file_name'属性
            image.path = dicom_path  # 为Image对象添加coco标签格式的'path'属性
            polygons = self._get_polygons(label_content_json)
            for class_name, polygon in polygons.items():
                mask = Mask.from_polygons(polygon)  # get mask
                categ = Category(class_name)
                categ.id = self.category_mapping(categ)
                image.add(mask, categ)
            dataset.add(image)
        with open(self.save_path, 'w') as output_json_file:  # 最后输出为json数据
            if dataset_type == 'coco':
                json.dump(dataset.coco(), output_json_file)
            elif dataset_type == 'voc':
                json.dump(dataset.voc(), output_json_file)


def parse_opt():
    parser = argparse.ArgumentParser(description='Cvt_Dataset')
    parser.add_argument('--excel_path', default='./examples/data/A2C_A4C/echo_2D_A4C_A2C_parameter_20220714.xlsx', type=str)
    parser.add_argument('--save_path', default='Forgery_test_4500.json', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    t = Lepu2COCO(excel_path=opt.excel_path, save_path=opt.save_path)
    t.get_dataset(dataset_type='coco')
