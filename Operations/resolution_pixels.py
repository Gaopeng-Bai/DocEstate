#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 7/23/2020 4:15 PM
# @Author  : Gaopeng.Bai
# @File    : resolution_pixels.py
# @User    : gaope
# @Software: PyCharm
# @Description:
# Reference:**********************************************
# coding=utf-8
import os  #
from PIL import Image
import re
import cv2


def resize_pixels(image, new_path):
    resize_width = 580
    resize_depth = 580

    im = Image.open(image)
    w, h = im.size
    if w < resize_width:
        h_new = int(resize_width * h / w)
        w_new = resize_width
        out = im.resize((w_new, h_new), Image.ANTIALIAS)
        out.save(new_path)
        repair(new_path)

    if h < resize_depth:
        h_new = int(resize_depth * w / h)
        w_new = resize_depth
        out = im.resize((h_new, w_new), Image.ANTIALIAS)
        out.save(new_path)
        repair(new_path)


def repair(images):
    im = cv2.imread(images)  # 读取图片rgb 格式<class 'numpy.ndarray'>
    image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 格式转换，bgr转rgb
    image.save(images, quality=95, dpi=(300.0, 300.0))  # 调整图像的分辨率为300,dpi可以更改
