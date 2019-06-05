# config.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict


# config for meta-learning
__C = edict()
cfg = __C

# Set some path
__C.voc0712_meta_info = './voc2017_metainfo'
__C.voc07test_meta_info = './voc2007test_metainfo'
__C.VOC0712_ROOT = './path to VOC2007 + VOC2012/VOCdevkit/'
__C.VOC07testVOC_ROOT = './path to VOC2007test/VOCdevkit/'

# idx for dataset (seen / unseen split)
__C.IDX = 1

# idx=1
if __C.IDX == 1:
    __C.seen_classes = ['pottedplant', 'tvmonitor', 'sofa',  'motorbike',
                        'horse', 'boat', 'dog', 'bicycle', 'train',
                        'sheep','bottle', 'person', 'aeroplane', 'diningtable', 'bird']
    __C.unseen_classes = ['cow', 'bus', 'cat', 'car', 'chair']

# idx=2
elif __C.IDX == 2:
    __C.seen_classes = ['pottedplant', 'tvmonitor', 'sofa', 'bus',
                         'boat',  'bicycle', 'train', 'cow', 'cat',
                        'car', 'chair', 'sheep', 'bottle',  'aeroplane',  'bird']
    __C.unseen_classes = ['diningtable', 'dog', 'horse', 'motorbike', 'person']

# idx=3
elif __C.IDX == 3:
    __C.seen_classes = ['bus', 'motorbike',
                        'horse', 'boat', 'dog', 'bicycle',  'cow', 'cat',
                        'car', 'chair', 'bottle', 'person', 'aeroplane', 'diningtable', 'bird']
    __C.unseen_classes = ['pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',]

# idx=4
elif __C.IDX == 4:
    __C.seen_classes = ['bus', 'motorbike', 'horse',  'dog',   'cow', 'cat',
                        'car', 'chair',  'person',  'diningtable', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor',]
    __C.unseen_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',]

else:
    raise NotImplementedError('No this idx {}'.format(__C.IDX))
# TODO other class partitions:
