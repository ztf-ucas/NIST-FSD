"""
NIST-FSD dataset by ztf-ucas
Reference: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from data.config import cfg
import os
import pickle

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = [  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

VOC_ROOT = cfg.VOC07testVOC_ROOT


def get_all_crop(target):
    """
    Arguments:
        target (annotation) : the target annotation to be made usable
            will be an ET.Element
    Returns:
        a list containing lists of bounding boxes  [bbox coords, class name]
    """
    res = []
    keep_difficult = False
    for obj in target.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if not keep_difficult and difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            bndbox.append(cur_pt)
        bndbox.append(name)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]

    return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOC07test_meta(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root=VOC_ROOT,
                 image_sets=[('2007', 'test')],
                 transform=None, target_transform=None,
                 dataset_name='VOC07test', idx=cfg.IDX, meta_idx=None, meta_classes=None):
        # idx: id for class partition
        # meta_idx: samples for the current task
        # meta_class: classes for the current task
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.idx = idx
        self.meta_idx = meta_idx
        self.meta_classes = meta_classes
        if self.meta_classes is None:
            self.classes = VOC_CLASSES
        else:  # meta
            self.classes = self.meta_classes
        if self.meta_idx is None:  # not meta
            for (year, name) in image_sets:
                rootpath = osp.join(self.root, 'VOC' + year)
                for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                    self.ids.append((rootpath, line.strip()))
        else:  # meta
            self.ids = self.meta_idx

        self.voc07test_meta_info = cfg.voc07test_meta_info
        if not osp.exists(self.voc07test_meta_info):
            os.mkdir(self.voc07test_meta_info)
        # get seen unseen classes
        self._get_seen_unseen_classes()

        # split images contain seen or unseen objects
        self.train_seen_pkl = osp.join(self.voc07test_meta_info, 'train_seen_' + str(self.idx) + '.pkl')
        self.train_unseen_pkl = osp.join(self.voc07test_meta_info, 'train_unseen_' + str(self.idx) + '.pkl')
        self.train_seen_unseen_pkl = osp.join(self.voc07test_meta_info, 'train_seen_unseen_' + str(self.idx) + '.pkl')
        if not osp.exists(self.train_seen_pkl):
            print('No seen_unseen split file, execute _filter_samples()')
            self._filter_samples()
        else:
            print('seen_unseen split file: ', self.train_seen_pkl)
        self.cls_dict_pkl = osp.join(self.voc07test_meta_info, 'cls_dict'+str(self.idx)+'.pkl')
        if not osp.exists(self.cls_dict_pkl):
            print('No cls_dict file, excute generate_cls_dict()')
            self.generate_cls_dict()
        else:
            print('cls_dict file: ', self.cls_dict_pkl)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height, self.classes)

        if self.transform is not None:
            target = np.array(target)
            # [[xmin, ymin, xmax, ymax, label_ind], ...]
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            # [[xmin, ymin, xmax, ymax, label_ind], ...]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _get_seen_unseen_classes(self):
        self.seen_classes = cfg.seen_classes
        self.unseen_classes = cfg.unseen_classes

    def _filter_samples(self):
        """
        train_seen: images contain only seen objects
        train_unseen: images contain only unseen objects
        train_seen_unseen: images contain seen and unseen objects
        """
        train_seen_names = []
        train_unseen_names = []
        train_seen_unseen_names = []
        name_list = self.ids
        for index, name in enumerate(name_list):
            xml_path = self._annopath % name
            target = ET.parse(xml_path).getroot()
            boxes = get_all_crop(target)
            cls_names = [box_info[-1] for box_info in boxes]
            # only seen class
            if self._check_if_clas_intersect(cls_names) == 's':
                train_seen_names.append(name)
            # only unseen class
            if self._check_if_clas_intersect(cls_names) == 'u':
                train_unseen_names.append(name)
            # only seen class
            if self._check_if_clas_intersect(cls_names) == 'su':
                train_seen_unseen_names.append(name)
        # seen (train+val)
        with open(os.path.join(self.voc0712_meta_info, 'train_seen_'+str(self.idx)+'.pkl'), 'wb') as f:
            pickle.dump(train_seen_names, f)
        # unseen (train+val)
        with open(os.path.join(self.voc0712_meta_info, 'train_unseen_' + str(self.idx) + '.pkl'), 'wb') as f:
            pickle.dump(train_unseen_names, f)
        # seen_unseen (train+val)
        with open(os.path.join(self.voc0712_meta_info, 'train_seen_unseen_' + str(self.idx) + '.pkl'), 'wb') as f:
            pickle.dump(train_seen_unseen_names, f)

    def _check_if_clas_intersect(self, cls_names):
        """
        s: this image contains only seen objects
        u: this image contains only unseen objects
        su: this image contains seen and unseen objects
        """
        seen_flag = True
        unseen_flag = True
        for cls_name in cls_names:
            if cls_name not in self.seen_classes:
                seen_flag = False
            if cls_name not in self.unseen_classes:
                unseen_flag = False

        if seen_flag:
            return 's'
        if unseen_flag:
            return 'u'
        else:
            return 'su'

    def generate_cls_dict(self):
        """
        {
        cls_name1: [idx1, idx2, ...],
        cls_name2: [idx1, idx2, ...],
        ...
        }
        """
        # process seen classes
        cls_dict = {}
        with open(self.train_seen_pkl, 'rb') as f:
            seen_idx_list = pickle.load(f)
        for cls_name in self.seen_classes:
            cls_dict[cls_name] = []

        for idx in seen_idx_list:
            xml_path = self._annopath % idx
            target = ET.parse(xml_path).getroot()
            boxes = get_all_crop(target)
            cls_names = [box_info[-1] for box_info in boxes]
            for cls_name in self.seen_classes:
                if cls_name in cls_names:
                    cls_dict[cls_name].append(idx)
        # process unseen classes
        with open(self.train_unseen_pkl, 'rb') as f:
            unseen_idx_list = pickle.load(f)
        for cls_name in self.unseen_classes:
            cls_dict[cls_name] = []
        for idx in unseen_idx_list:
            xml_path = self._annopath % idx
            target = ET.parse(xml_path).getroot()
            boxes = get_all_crop(target)
            cls_names = [box_info[-1] for box_info in boxes]
            for cls_name in self.unseen_classes:
                if cls_name in cls_names:
                    cls_dict[cls_name].append(idx)
        # save cls_dict into disk
        with open(self.cls_dict_pkl, 'wb') as f:
            pickle.dump(cls_dict, f)



if __name__ == '__main__':
    db = VOC07test_meta(root=VOC_ROOT,transform=None)
    print(db.seen_classes)
    print(db.unseen_classes)



