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


def get_roi(ds):
    series = ds.pixel_array  # 像素值矩阵
    if not ds['PhotometricInterpretation'].value == 'RGB':
        series = convert_color_space(series, ds['PhotometricInterpretation'].value, 'RGB', per_frame=True)
    if len(series.shape) == 2:
        series = np.expand_dims(series, axis=0)
    if len(series.shape) == 3:
        series = np.expand_dims(series, axis=series.shape[-1] != 3 and 3)
    ROI_info = ds.SequenceOfUltrasoundRegions[0]
    x_max, y_max, x_min, y_min = ROI_info.RegionLocationMaxX1, ROI_info.RegionLocationMaxY1, ROI_info.RegionLocationMinX0, ROI_info.RegionLocationMinY0
    series = series[:, y_min:y_max, x_min:x_max, :]
    # zero padding to get a square matrix, ((extra_top, extra_bottom), (extra_left, extra_right), (0, 0))
    frames, height, width, channels = series.shape
    how_to_padding = ((0, 0), (0, 0), (0, height - width), (0, 0)) if height > width else (
        (0, 0), (0, width - height), (0, 0), (0, 0))
    series = np.pad(series, how_to_padding, mode='constant', constant_values=0)
    return series


def make_video(file_path, destinationFolder, cropSize=(112, 112)):
    try:
        ds = pydicom.read_file(file_path)  # DICOM文件的位置
        series = get_roi(ds)  # N,H,W,C( frames, height, width, channels)
        # TODO should add anno change
        try:
            fps = ds[(0x18, 0x40)].value
        except:
            fps = 30
            print("couldn't find frame rate, default to 30")
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        video_filename = os.path.join(destinationFolder + file_path + '.avi')
        assert is_pathname_valid(video_filename), 'wrong file path'
        # os.makedirs(os.path.split(video_filename)[0], exist_ok=True)
        pathlib.Path(os.path.split(video_filename)[0]).mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(video_filename, fourcc, fps, cropSize)
        for slice in series:
            if slice.shape[-1] == 1:
                slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
            # Resize image
            slice = cv2.resize(slice, cropSize, interpolation=cv2.INTER_CUBIC)
            out.write(slice)
        out.release()
    except:
        print("something filed, not sure what, have to debug", file_path)
    return 0


class PrepareData(object):
    def __init__(self, data_path, dst_dir, split, crop_size, scaling_factor):
        self.data_frame = pd.read_excel(data_path)
        """
        :参数 data_path: # 数据文件路径
        :参数 split: 'train' 或者 'test'
        :参数 crop_size: 高分辨率图像裁剪尺寸  （实际训练时不会用原图进行放大，而是截取原图的一个子块进行放大）
        :参数 scaling_factor: 放大比例
        :参数 test_data_name: 如果是评估阶段，则需要给出具体的待评估数据集名称，例如 "Set14"
        """
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.dst_folder = dst_dir
        # 如果是训练，则所有图像必须保持固定的分辨率以此保证能够整除放大比例
        # 如果是测试，则不需要对图像的长宽作限定
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "裁剪尺寸不能被放大比例整除!"
        self.data_frame.rename(
            columns={'FilePath': 'FileName', 'Rows': 'FrameHeight', 'Columns': 'FrameWidth', 'FrameIndex': 'Frame',
                     'train_test': 'Split'}, inplace=True)

    def __len__(self):
        return len(self.data_frame)

    def get_filelist(self, path_or_buf):
        df = self.data_frame.drop_duplicates(subset=['FilePath'])
        df.to_csv(path_or_buf=path_or_buf, encoding='gbk', index=False,
                  columns=['type', 'FileName', 'NumberOfFrames', 'FrameHeight', 'FrameWidth', 'View',
                           'LabelContent', 'Frame', 'Quality', 'Regions', 'label', 'A2C_A4C', 'Split'])

    def get_volume_tracings(self, path_or_buf):
        df = self.data_frame
        tags = ['1.5.1', '1.5.2', '5.1.5', '5.1.7', '1.5.3', '1.5.4', '5.1.1', '5.1.3', '1.5.5', '1.5.6', '5.1.6',
                '5.1.8', '1.5.7', '1.5.8', '5.1.2', '5.1.4']
        tags_def = ['A4C_DIA'] * 4 + ['A4C_SYS'] * 4 + ['A2C_DIA'] * 4 + ['A2C_SYS'] * 4
        tags = dict(zip(tags, tags_def))
        left_ventricular = ['1.5.1', '1.5.2', '1.5.3', '1.5.4', '1.5.5', '1.5.6', '1.5.7', '1.5.8']  # 左心室
        left_atrium = ['5.1.1', '5.1.2', '5.1.3', '5.1.4', '5.1.4', '5.1.6', '5.1.7', '5.1.8']  # 左心房

        def get_merge_value(label_content_json):
            left_ventricular_points_x, left_ventricular_points_y, left_atrium_points_x, left_atrium_points_y = [], [], [], []
            label_content = json.loads(label_content_json)
            for region in label_content['regions']:
                if region['shape_attributes']['name'] == 'dotted' and region['region_attributes'][0][
                    'type'] in left_ventricular:
                    left_ventricular_points_x, left_ventricular_points_y = region['shape_attributes'][
                                                                               'all_points_x'], \
                                                                           region['shape_attributes'][
                                                                               'all_points_y']
                elif region['shape_attributes']['name'] == 'dotted' and region['region_attributes'][0][
                    'type'] in left_atrium:
                    left_atrium_points_x, left_atrium_points_y = region['shape_attributes'][
                                                                     'all_points_x'], \
                                                                 region['shape_attributes'][
                                                                     'all_points_y']
            # assert len(mask_regions) == 2, 'wrong mask_regions length, should be 2, got %d' % len(mask_regions)
            points = list(zip_longest(left_ventricular_points_x, left_ventricular_points_y, left_atrium_points_x,
                                      left_atrium_points_y, fillvalue=None))
            return points

        df['merge'] = df.apply(lambda x: get_merge_value(x['LabelContent']), axis=1)
        df_exploded = df.explode('merge')
        df_exploded['X1'], df_exploded['Y1'], df_exploded['X2'], df_exploded['Y2'] = df_exploded['merge'].str
        df_exploded.to_csv(path_or_buf=path_or_buf, encoding='gbk', index=False)

    def get_dataset(self):
        for index, file_path in enumerate(self.data_frame['FilePath']):
            if index == 10:
                break
            if not os.path.exists(os.path.join(self.dst_folder, file_path + ".avi")):
                make_video(file_path, self.dst_folder)
            else:
                print("Already did this file", file_path)


if __name__ == "__main__":
    dst_dir = '/home/qingyu_zhao/data/code/AnalyseDicom/datasets/echonet_dynamic_avi_lepu'
    # source_file = '/data/share/echo/PLAX/echo_2D_plax_8_anyone_parameter_20220624.xlsx'
    source_file = './A2C_A4C/echo_2D_A4C_A2C_parameter_20220714.xlsx'
    d = PrepareData(source_file,
                    dst_dir,
                    split='train',
                    crop_size=96,
                    scaling_factor=4)
    # d.get_dataset()
    # d.get_filelist('./data/FileList_test.csv')
    d.get_volume_tracings('./data/VolumeTracings_test.csv')
    pass
