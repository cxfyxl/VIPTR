# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np
import random
from PIL import Image
from dataload.aug import tia_distort, tia_stretch, tia_perspective

def ImgPIL2CV(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img


def ImgPIL2CV_fast(img):
    img = np.array(img2)[:, :, ::-1]
    return img


def ImgCV2PIL(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img


class RecConAug(object):
    def __init__(self,
                 prob=0.5,
                 image_shape=(32, 320, 3),
                 max_text_length=25,
                 ext_data_num=1,
                 **kwargs):
        self.ext_data_num = ext_data_num
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(data['image'].shape[1] / data['image'].shape[0] * self.image_shape[0])
        ext_w = round(ext_data['image'].shape[1] / ext_data['image'].shape[0] * self.image_shape[0])
        data['image'] = cv2.resize(data['image'], (ori_w, self.image_shape[0]))
        ext_data['image'] = cv2.resize(ext_data['image'], (ext_w, self.image_shape[0]))
        data['image'] = np.concatenate([data['image'], ext_data['image']], axis=1)
        data["label"] += ext_data["label"]
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data["ext_data"]):
            if len(data["label"]) + len(ext_data["label"]) > self.max_text_length:
                break
            concat_ratio = data['image'].shape[1] / data['image'].shape[0] + ext_data['image'].shape[1] / \
                           ext_data['image'].shape[0]
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop("ext_data")
        return data


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 5
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


def resizeAug(img, HThresh=25):
    if img.shape[0] > HThresh:
        h, w = img.shape[:2]
        ratio = np.random.randint(4, 8) / 10.
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        img = cv2.resize(img, (w, h))
    return img


def random_dilute(image, sele_value=50, set_value=20, num_ratio=[0.1, 0.2]):
    index = np.where(image < (image.min() + sele_value))
    tag = []
    for i in range(len(index[0])):
        tag.append([index[0][i], index[1][i]])
    np.random.shuffle(tag)
    tag = np.array(tag)
    total_num = len(tag[:, 0])
    num = int(total_num * np.random.choice(num_ratio, 1)[0])
    tag1 = tag[:num, 0]
    tag2 = tag[:num, 1]
    index = (tag1, tag2)
    start = image.min() + sele_value + set_value
    if (start >= 230):
        start = start - 50
    ra_value = min(np.random.randint(min(225, start), 230), 230)
    image[index] = ra_value
    return image


def motion_blur(image):
    # degree建议：2 - 5
    # angle建议：0 - 360
    # 都为整数
    degree = np.random.randint(2, 6)
    angle = np.random.randint(0, 360)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred_image = np.array(blurred, dtype=np.uint8)
    return blurred_image


class Config:
    """
    Config
    """

    def __init__(self, use_tia):
        self.anglex = random.random() * 30
        self.angley = random.random() * 15
        self.anglez = random.random() * 10
        self.fov = 42
        self.r = 0
        self.shearx = random.random() * 0.3
        self.sheary = random.random() * 0.05
        self.borderMode = cv2.BORDER_REPLICATE
        self.use_tia = use_tia

    def make(self, w, h, ang):
        """
        make
        """
        self.anglex = random.random() * 5 * flag()
        self.angley = random.random() * 5 * flag()
        self.anglez = -1 * random.random() * int(ang) * flag()
        self.fov = 42
        self.r = 0
        self.shearx = 0
        self.sheary = 0
        self.borderMode = cv2.BORDER_REPLICATE
        self.w = w
        self.h = h

        self.perspective = self.use_tia
        self.stretch = self.use_tia
        self.distort = self.use_tia

        self.crop = False  # True
        self.affine = False  # False
        self.reverse = False
        self.noise = True
        self.jitter = False  # True
        self.blur = True
        self.color = False  # True
        self.random_dilute = False  # True


def rad(x):
    """
    rad
    """
    return x * np.pi / 180


def get_warpR(config):
    """
    get_warpR
    """
    anglex, angley, anglez, fov, w, h, r = \
        config.anglex, config.angley, config.anglez, config.fov, config.w, config.h, config.r
    if w > 69 and w < 112:
        anglex = anglex * 1.5

    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
    # Homogeneous coordinate transformation matrix
    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0], [
                       0,
                       -np.sin(rad(anglex)),
                       np.cos(rad(anglex)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0], [
                       -np.sin(rad(angley)),
                       0,
                       np.cos(rad(angley)),
                       0,
                   ], [0, 0, 0, 1]], np.float32)
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    r = rx.dot(ry).dot(rz)
    # generate 4 points
    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)
    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
    list_dst = np.array([dst1, dst2, dst3, dst4])
    org = np.array([[0, 0], [w, 0], [0, h], [w, h]], np.float32)
    dst = np.zeros((4, 2), np.float32)
    # Project onto the image plane
    dst[:, 0] = list_dst[:, 0] * z / (z - list_dst[:, 2]) + pcenter[0]
    dst[:, 1] = list_dst[:, 1] * z / (z - list_dst[:, 2]) + pcenter[1]

    warpR = cv2.getPerspectiveTransform(org, dst)

    dst1, dst2, dst3, dst4 = dst
    r1 = int(min(dst1[1], dst2[1]))
    r2 = int(max(dst3[1], dst4[1]))
    c1 = int(min(dst1[0], dst3[0]))
    c2 = int(max(dst2[0], dst4[0]))

    try:
        ratio = min(1.0 * h / (r2 - r1), 1.0 * w / (c2 - c1))

        dx = -c1
        dy = -r1
        T1 = np.float32([[1., 0, dx], [0, 1., dy], [0, 0, 1.0 / ratio]])
        ret = T1.dot(warpR)
    except:
        ratio = 1.0
        T1 = np.float32([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ret = T1
    return ret, (-r1, -c1), ratio, dst


def get_warpAffine(config):
    """
    get_warpAffine
    """
    anglez = config.anglez
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0]], np.float32)
    return rz


def warp(img, use_tia=True, prob=0.4, angz=2):
    """
    warp
    """
    h, w, _ = img.shape
    config = Config(use_tia=use_tia)
    config.make(w, h, angz)
    new_img = img

    if config.distort:
        img_height, img_width = img.shape[0:2]
        if random.random() <= (prob - 0.1) and img_height >= 32 and img_width >= 32:
            new_img = tia_distort(new_img, 10)

    if config.stretch:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 32 and img_width >= 32:
            new_img = tia_stretch(new_img, 10)

    if config.perspective:
        if random.random() <= prob:
            new_img = tia_perspective(new_img)

    if config.crop:
        img_height, img_width = img.shape[0:2]
        if random.random() <= prob and img_height >= 32 and img_width >= 32:
            new_img = get_crop(new_img)

    if config.blur:
        bprob = prob / 3.
        if random.random() <= bprob:
            new_img = blur(new_img)
        elif random.random() > bprob and random.random() <= 2 * bprob:
            new_img = resizeAug(new_img)
        elif random.random() > 2 * bprob and random.random() <= 3 * bprob:
            new_img = motion_blur(new_img)

    if config.random_dilute:
        if random.random() <= prob:
            new_img = random_dilute(new_img)

    if config.color:
        if random.random() <= prob:
            new_img = cvtColor(new_img)
    if config.jitter:
        new_img = jitter(new_img)
    if config.noise:
        if random.random() <= prob:
            new_img = add_gasuss_noise(new_img)
    if config.reverse:
        if random.random() <= prob:
            new_img = 255 - new_img

    return new_img

