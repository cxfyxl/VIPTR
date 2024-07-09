# -*- coding: utf-8 -*-

import os
import lmdb
import sys
import six
import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataload.dataAug import *
import torchvision.transforms as transforms
from utils.label2tensor import strLabelConverter

def Add_Padding(image, top, bottom, left, right, color=(255,255,255)):
    if(not isinstance(image,np.ndarray)):
        image = np.array(image)
    padded_image = cv2.copyMakeBorder(image, top, bottom,left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

def fixkeyCh(key):
    return ''.join(re.findall("[㙟\u4e00-\u9fa50-9a-zA-Z#%().·-]",key))

def fixkeyEn(key):
    return ''.join(re.findall("[0-9a-zA-Z]",key))

class LoadDatasetLmdb(Dataset):
    def __init__(self,config,lmdb_file):
        num_workers = config['train']['num_workers']
        self.fixKey = config['train']['fixKeyON']
        self.fixKeyType = config['train']['fixKeytype']
        assert self.fixKeyType in ['En','Ch']
        self.env = lmdb.open(lmdb_file, max_readers=num_workers, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % (lmdb_file))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')))
            self.nSamples = nSamples
        

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8')).decode().replace('\ufeff', '').replace('\u3000', '').strip()
        if self.fixKey:
            if self.fixKeyType == 'En':
                label = fixkeyEn(label)
                label = label.lower()
            elif self.fixKeyType == 'Ch':
                label = fixkeyCh(label)
        return (img, label)


class resizeNormalize(object):
    def __init__(self, height=32, max_width=280, types='train'):
        assert types in ['train','val','test']
        self.toTensor = transforms.ToTensor()
        self.max_width = max_width
        self.types = types
        self.height = height
    def __call__(self, img):
        if (self.types == 'train' or self.types == 'val'):
            w, h = img.size
            img = img.resize((int(self.height / float(h) * w), self.height), Image.BILINEAR)
            w, h = img.size
            if (w < self.max_width):
                img = Add_Padding(img, 0, 0, 0, self.max_width - w)
                img = Image.fromarray(img)
            else:
                img = img.resize((self.max_width, self.height), Image.BILINEAR)
        elif self.types == 'test':
            w, h = img.size
            img = img.resize((int(self.height / float(h) * w)//4*4, self.height), Image.BILINEAR)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img

class alignCollate(object):
    def __init__(self, config,trans_type):
        self.imgH = config['train']['imgH']
        self.imgW = config['train']['imgW']
        self.use_tia = config['train']['use_tia']
        self.aug_prob = config['train']['aug_prob']
        self.label_transform = strLabelConverter(config['train']['alphabet'])
        self.trans_type = trans_type
        self.isGray = config['train']['isGray']
        self.ConAug = config['train']['ConAug']
        
    def __call__(self, batch):
        images, labels = zip(*batch)
        new_images = []
        for (image,label) in zip(images,labels):
            if self.trans_type == 'train':
                
#                 image = np.array(image)
#                 try:
#                     image = warp(image,self.use_tia,self.aug_prob)
#                 except:
#                     pass
#                 image = Image.fromarray(image)
                
                if self.isGray:
                    image = image.convert('L')
            new_images.append(image)
        transform = resizeNormalize(self.imgH, self.imgW, self.trans_type)
        
        fix_image = []
        fix_label = []
        for (img,label) in zip(new_images,labels):
            try:
                img = transform(img)
                fix_image.append(img)
                fix_label.append(label)
            except:
                pass
        fix_image = torch.cat([t.unsqueeze(0) for t in fix_image], 0)
        intText,intLength = self.label_transform.encode(fix_label)
        return fix_image, intText,intLength,fix_label

def CreateDataset(config,lmdb_type):
    assert lmdb_type in ['train','val']
    if lmdb_type == 'train':
        lmdb_file = config['train']['train_lmdb_file']
        assert isinstance(lmdb_file,list)
        assert len(lmdb_file)>=1
        train_dataset = LoadDatasetLmdb(config,os.path.join(config['train']['data_root_train'],lmdb_file[0]))
        for i in range(1,len(lmdb_file)):
            train_dataset+=LoadDatasetLmdb(config,os.path.join(config['train']['data_root_train'],lmdb_file[i]))
        return train_dataset
    elif lmdb_type == 'val':
        lmdb_file = config['train']['val_lmdb_file']
        val_datasets = []
        for i in range(len(lmdb_file)):
            val_datasets.append(LoadDatasetLmdb(config,os.path.join(config['train']['data_root_val'],lmdb_file[i])))
        return val_datasets
        
    