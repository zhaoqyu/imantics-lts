import os, json, numpy as np
from tqdm import tqdm
from imantics_lts import Mask, Image, Category, Dataset
import cv2

dataset = Dataset('forgery') # 先定义一个数据库对象，后续需要往里面添加具体的image和annotation

path = 'test/mask/' # image对应的mask的文件路径
for index, i in enumerate(tqdm(os.listdir(path))):
    mask_file = os.path.join(path, i)
    name = i.split('_')[0]
    file = os.path.join(path, '../img/', '{}.jpg'.format(name))

    image = cv2.imread(file)[:,:,::-1]
    image = Image(image, id=index+1) # 定义一个Image对象
    image.file_name = '{}.jpg'.format(name) # 为上面的Image对象添加coco标签格式的'file_name'属性
    image.path = file # 为Image对象添加coco标签格式的'path'属性

    mask = cv2.imread(mask_file, 0)
    t = cv2.imread(file)
    if t.shape[:-1] != mask.shape:
        h, w, _ = t.shape
        mask = cv2.resize(mask, (w, h), cv2.INTER_CUBIC)

    mask = Mask(mask) # 定义一个Mask对象，并传入上面所定义的image对应的mask数组

    categ = i.split('_')[1]
    t = Category(categ) # 这里是定义Category对象
    if categ == 'splice':
        t.id = 1 # 手动指定类别id号
    elif categ == 'copymove':
        t.id = 2 # 同上
    elif categ == 'removal':
        t.id = 3
    image.add(mask, t) # 将mask信息和类别信息传给image

    dataset.add(image) # 往dataset里添加图像以及gt信息

t = dataset.coco() # 将dataset转化为coco格式的，还可以转化为yolo等格式
with open('Forgery_test_4500.json', 'w') as output_json_file: # 最后输出为json数据
    json.dump(t, output_json_file)
