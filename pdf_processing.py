#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 7/21/2020 2:20 PM
# @Author  : Gaopeng.Bai
# @File    : pdf_processing.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************
import os
import platform
import shutil
import cv2
import numpy as np
from Operations.pdf_to_images import pyMuPDF_fitz
from Operations.Recognition import ocr_core
from Operations.excel_operation import excel_operator
from Operations.resolution_pixels import resize_pixels


system = platform.system()
def UsePlatform():
    if system == "Windows":

        print("Call Windows tasks")
    elif system == "Linux":
        print("Call Linux tasks")
    else:
        print("Other System tasks")


def width_calculate(image):
    img = cv2.imread(image)
    height, width = img.shape[:2]
    return height, width, img


def DelOtherLines(MyList):
    ListLen = len(MyList)
    NewList = []
    for i in range(ListLen):
        x1, x2 = MyList[i][0], MyList[i][2]
        if abs(x1 - x2) < 20:
            NewList.append(MyList[i])

    return NewList


def cut_area(img, i, x, y, h, w, path):
    new_image = img[y + 2:y + h - 2, x + 2:x + w - 2]  #

    if not os.path.isdir(path):
        os.makedirs(path)
    cv2.imwrite(path + str(i) + ".png", new_image)
    if "reg" not in path:
        delete_lines(path + str(i) + ".png")
        print("Delete the lines in the image")


def delete_lines(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    cnts = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
    for c in cnts2:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    cv2.imwrite(img_path, img=image)


def count_height(image):
    img = cv2.imread(image, 0)
    height, width = img.shape[:2]

    # Horizontal projection count the number of black dots in each row
    horizontal = np.zeros(height, dtype=np.int32)
    for y in range(height, 0):
        for x in range(0, width):
            if img[y, x] != 255:
                horizontal[y] += 1

    for i in range(height, 0):
        if horizontal[i] > 0:  # Enter the text area from the blank area
            return i


def rows_cut(image, gap=20):
    img = cv2.imread(image, 0)
    height, width = img.shape[:2]
    height_cut = np.zeros((height, 2), dtype=np.int32)

    # Horizontal projection count the number of black dots in each row
    horizontal = np.zeros(height, dtype=np.int32)
    for y in range(0, height):
        for x in range(0, width):
            if img[y, x] != 255:
                horizontal[y] += 1
    # Select the row split point according to the horizontal projection value
    inline = 1
    start = 0
    j = 0
    for i in range(0, height):
        if inline == 1 and horizontal[i] > 0:  # Enter the text area from the blank area
            start = i  # Record starting line split point
            inline = 0
        elif (i - start > gap) and horizontal[i] <= 0 and inline == 0:  # Enter the blank area from the text area
            inline = 1
            if start > 10:
                height_cut[j][0] = start - 10  # Save row split position
            else:
                height_cut[j][0] = 0

            height_cut[j][1] = i + 2
            j = j + 1

    cut_area = []
    cut_area2 = []
    for i in range(len(height_cut)):
        if height_cut[i][1] != 0:
            y0 = height_cut[i][0]
            y1 = height_cut[i][1]
            cut_area.append(y0)
            cut_area2.append(y1)

    return cut_area


def reload_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)


class pdf_processing:

    def __init__(self, pdf='../data_/38_GBA/Freudenstadt.pdf'):

        self.images = 'images'
        self.pdfPath = pdf
        self.cut_dir = "images/test/"

        self.BeS_dir = os.path.join(self.cut_dir, "Bestandsverzeichnis/")
        self.Erste_dir = os.path.join(self.cut_dir, "ErsteAbteilung/")
        self.Zweite_dir = os.path.join(self.cut_dir, "ZweiteAbteilung/")
        self.Dritte1_dir = os.path.join(self.cut_dir, "Dritte1Abteilung/")
        self.Dritte2_dir = os.path.join(self.cut_dir, "Dritte2Abteilung/")
        self.dritte2 = False

        self.reg_dir = os.path.join(self.cut_dir, "reg/")

        self.load_files()

    def pdf_to_images(self, pdf):
        # clear all images
        reload_dir(self.images)

        pyMuPDF_fitz(pdf, self.images)
        self.load_files()

    def load_files(self):
        self.image_files = []
        if os.path.exists(self.images):
            file_suffix = '.png'
            if os.path.isfile(self.images):
                if os.path.splitext(self.images)[1] == file_suffix:
                    self.image_files.append(self.images)

            elif os.path.isdir(self.images):
                for file in os.listdir(self.images):
                    file_path = os.path.join(self.images, file)
                    # print(file)
                    # print(os.path.splitext(file_path)[1] == 'pdf')
                    if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == file_suffix:
                        self.image_files.append(file_path)
        else:
            os.mkdir(self.images)

    def remove_no_table_images(self):
        file_name = []
        allTestDataName = []
        for filename in os.listdir(self.images):
            if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                # print(filename)
                allTestDataName.append(filename)

        allTestDataName.sort(key=lambda x: int(x[:-4]))

        for file in allTestDataName:
            if system == "Windows":
                file = self.images + "\\" + file
            else:
                file = self.images + "/" + file
            data = ocr_core(file)
            if not ("Bestandsverzeichnis" in data or "Bastandsvarzeichnis" in data or
                    "Erste Abteilung" in data or
                    "Zweite Abteilung" in data or
                    "Dritte Abteilung" in data or "Dritta Abteilung" in data or "Dritts Abteilung" in data):
                os.remove(file)
        # self.load_files()
        allTestDataName = []
        for filename in os.listdir(self.images):
            if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                # print(filename)
                allTestDataName.append(filename)

        allTestDataName.sort(key=lambda x: int(x[:-4]))
        a = 0
        for file in allTestDataName:
            if system == "Windows":
                file = self.images + "\\" + file
                old_file_name = file.split("\\")[1].split(".png")[0]
            else:
                file = self.images + "/" + file
                old_file_name = file.split("/")[1].split(".png")[0]

            file_name.append(int(old_file_name))
            a = min(file_name)
        cache_file = []
        cache_newName = []
        for file in allTestDataName:
            if system == "Windows":
                file = self.images + "\\" + file
                old_file_name = file.split("\\")[1].split(".png")[0]
                new_name = file.split("\\")[0] + "\\" + str(int(old_file_name) - a) + ".png"
            else:
                file = self.images + "/" + file
                old_file_name = file.split("/")[1].split(".png")[0]
                new_name = file.split("/")[0] + "/" + str(int(old_file_name) - a) + ".png"

            if int(old_file_name) > 9:
                cache_file.append(file)
                cache_newName.append(new_name)
            else:
                os.rename(file, new_name)
        for i, cfile in enumerate(cache_file):
            os.rename(cfile, cache_newName[i])

    def Processing(self):
        page = 0
        self.load_files()
        allTestDataName = []
        for filename in os.listdir(self.images):
            if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                # print(filename)
                allTestDataName.append(filename)

        allTestDataName.sort(key=lambda x: int(x[:-4]))

        for file in allTestDataName:
            if system == "Windows":
                file = self.images + "\\" + file
                current_page = int(file.split("\\")[1].split(".png")[0])
            else:
                file = self.images + "/" + file
                current_page = int(file.split("/")[1].split(".png")[0])

            data = ocr_core(file)
            print(file)
            img = cv2.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(image=binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            conn = {"x": [], "y": [], "w": [], "h": []}
            a = []
            for i in range(0, len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                if h > 100:
                    a.append(h)
                if 900 > h >= 860:
                    print("h:{}, x: {}, y:{}, w: {}".format(h, x, y, w))
                    conn["x"].append(x)
                    conn["y"].append(y)
                    conn["w"].append(w)
                    conn["h"].append(h)
                    # cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 3)
            if len(conn["x"]) == 0:
                a = []
                for i in range(0, len(contours)):
                    x, y, w, h = cv2.boundingRect(contours[i])
                    if h > 100:
                        a.append(h)
                    if 665 > h >= 570:
                        print("h:{}, x: {}, y:{}, w: {}".format(h, x, y, w))
                        conn["x"].append(x)
                        conn["y"].append(y)
                        conn["w"].append(w)
                        conn["h"].append(h)
            # test table located.

            #             cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 3)
            # cv2.imshow('test', img)
            # cv2.waitKey(0)

            if "Bestandsverzeichnis" in data or "Bastandsvarzeichnis" in data:

                if current_page == 0:
                    # 1-4 Bestandsverzeichnis
                    print("processing Bestandsverzeichnis first page")
                    if len(conn["x"]) == 8:
                        temp_w = conn["w"][3] + conn["w"][4] + conn["w"][5]
                        for i, value in enumerate(conn["x"]):
                            if 3 <= i <= 5:
                                pass
                            elif i < 3:
                                cut_area(img, len(conn["x"]) - i - 2, value, conn["y"][i], conn["h"][i],
                                         conn["w"][i], path=self.BeS_dir)
                            else:
                                cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                         conn["w"][i], path=self.BeS_dir)
                        cut_area(img, i=3, x=conn["x"][5], y=conn["y"][5], h=conn["h"][5], w=temp_w, path=self.BeS_dir)
                    elif len(conn["x"]) == 7:
                        temp_w = conn["x"][2] - conn["x"][4]
                        for i, value in enumerate(conn["x"]):
                            if 3 <= i <= 4:
                                pass
                            elif i < 3:
                                cut_area(img, len(conn["x"]) - i - 1, value, conn["y"][i], conn["h"][i],
                                         conn["w"][i], path=self.BeS_dir)
                            else:
                                cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                         conn["w"][i], path=self.BeS_dir)
                        cut_area(img, i=3, x=conn["x"][4], y=conn["y"][4], h=conn["h"][4], w=temp_w, path=self.BeS_dir)

                    else:
                        for i, value in enumerate(conn["x"]):
                            cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                     conn["w"][i], path=self.BeS_dir)
                else:
                    print("processing Bestandsverzeichnis second page")
                    for i, value in enumerate(conn["x"]):
                        cut_area(img, len(conn["x"]) - i + 6, value, conn["y"][i], conn["h"][i],
                                 conn["w"][i], path=self.BeS_dir)

            elif "Erste Abteilung" in data:
                if current_page == 2:
                    print("processing Erste Abteilung first page")
                    for i, value in enumerate(conn["x"]):
                        cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                 conn["w"][i], path=self.Erste_dir)
                else:
                    print("processing Erste Abteilung second page")
                    for i, value in enumerate(conn["x"]):
                        cut_area(img, len(conn["x"]) - i + 4, value, conn["y"][i], conn["h"][i],
                                 conn["w"][i], path=self.Erste_dir)

            elif "Zweite Abteilung" in data:
                if len(conn["x"]) < 4:
                    print("processing Zweite Abteilung first page")
                    for i, value in enumerate(conn["x"]):
                        cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                 conn["w"][i], path=self.Zweite_dir)
                if len(conn["x"]) >= 4:
                    print("processing Zweite Abteilung second page")
                    for i, value in enumerate(conn["x"]):
                        cut_area(img, len(conn["x"]) - i + 3, value, conn["y"][i], conn["h"][i],
                                 conn["w"][i], path=self.Zweite_dir)

            elif "Dritte Abteilung" in data or "Dritta Abteilung" in data or "Dritts Abteilung" in data:
                if current_page < 8:
                    print("processing Dritte 1 Abteilung first page")
                    page += 1
                    if page == 1:
                        print("processing Dritte 1 Abteilung first page")
                        for i, value in enumerate(conn["x"]):
                            cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                     conn["w"][i], path=self.Dritte1_dir)
                    else:
                        print("processing Dritte 1 Abteilung second page")
                        for i, value in enumerate(conn["x"]):
                            cut_area(img, len(conn["x"]) - i + 4, value, conn["y"][i], conn["h"][i],
                                     conn["w"][i], path=self.Dritte1_dir)
                else:
                    self.dritte2 = True
                    page += 1
                    if page == 3:
                        print("processing Dritte 2 Abteilung first page")
                        for i, value in enumerate(conn["x"]):
                            cut_area(img, len(conn["x"]) - i, value, conn["y"][i], conn["h"][i],
                                     conn["w"][i], path=self.Dritte2_dir)
                    else:
                        print("processing Dritte 2 Abteilung second page")
                        for i, value in enumerate(conn["x"]):
                            cut_area(img, len(conn["x"]) - i + 4, value, conn["y"][i], conn["h"][i],
                                     conn["w"][i], path=self.Dritte2_dir)

    def split_rows(self, flag="Bestandsverzeichnis", gap=20):
        reload_dir(self.reg_dir)
        self.rows2 = 0
        if flag == "Bestandsverzeichnis":
            allTestDataName = []
            for filename in os.listdir(self.BeS_dir[0:-1]):
                if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                    # print(filename)
                    allTestDataName.append(filename)

            allTestDataName.sort(key=lambda x: int(x[:-4]))

            split_flag = rows_cut(self.BeS_dir + str(1) + ".png", gap=40)
            self.rows = len(split_flag)

            for i, image in enumerate(allTestDataName):
                for j, value in enumerate(split_flag):
                    # if j >= 6:
                    #     break
                    height, width, img = width_calculate(self.BeS_dir + image)
                    # Pixel fine-tuning
                    gray = cv2.imread(self.BeS_dir + image, 0)
                    for x in range(0, width):
                        while gray[value, x] != 255:
                            if value == -1:
                                break
                            value -= 1
                    # cut elements by rows.
                    if j >= len(split_flag) - 1:
                        cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                    else:
                        cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag[j + 1] - value, w=width,
                                 path=self.reg_dir)
            print("split rows Bestandsverzeichnis done")

        if flag == "Erste Abteilung":
            allTestDataName = []
            for filename in os.listdir(self.Erste_dir[0:-1]):
                if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                    # print(filename)
                    allTestDataName.append(filename)

            allTestDataName.sort(key=lambda x: int(x[:-4]))

            for i, image in enumerate(allTestDataName):
                if i < 5:
                    split_flag = rows_cut(self.Erste_dir + str(1) + ".png", gap=gap)
                    self.rows = len(split_flag)
                    for j, value in enumerate(split_flag):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Erste_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Erste_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag[j + 1] - value, w=width,
                                     path=self.reg_dir)
                elif i >= 5:
                    split_flag2 = rows_cut(self.Erste_dir + str(8) + ".png", gap=gap)
                    self.rows2 = len(split_flag2)
                    for j, value in enumerate(split_flag2):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Erste_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Erste_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag2) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag2[j + 1] - value, w=width,
                                     path=self.reg_dir)
            print("split rows Erste Abteilung done")

        if flag == "Zweite Abteilung":
            allTestDataName = []
            for filename in os.listdir(self.Zweite_dir[0:-1]):
                if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                    # print(filename)
                    allTestDataName.append(filename)

            allTestDataName.sort(key=lambda x: int(x[:-4]))

            for i, image in enumerate(allTestDataName):
                if i < 3:
                    split_flag = rows_cut(self.Zweite_dir + str(1) + ".png", gap=gap)
                    self.rows = len(split_flag)
                    for j, value in enumerate(split_flag):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Zweite_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Zweite_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag[j + 1] - value, w=width,
                                     path=self.reg_dir)
                else:
                    split_flag2 = rows_cut(self.Zweite_dir + str(7) + ".png", gap=50)
                    self.rows2 = len(split_flag2)
                    for j, value in enumerate(split_flag2):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Zweite_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Zweite_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag2) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag2[j + 1] - value, w=width,
                                     path=self.reg_dir)

            print("split rows Zweite Abteilung done")

        if flag == "Dritte1 Abteilung":
            allTestDataName = []
            for filename in os.listdir(self.Dritte1_dir[0:-1]):
                if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                    # print(filename)
                    allTestDataName.append(filename)

            allTestDataName.sort(key=lambda x: int(x[:-4]))

            for i, image in enumerate(allTestDataName):
                if i < 4:
                    split_flag = rows_cut(self.Dritte1_dir + str(1) + ".png", gap=gap)
                    self.rows = len(split_flag)
                    for j, value in enumerate(split_flag):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Dritte1_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Dritte1_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag[j + 1] - value, w=width,
                                     path=self.reg_dir)
                elif i >= 4:
                    split_flag2 = rows_cut(self.Dritte1_dir + str(10) + ".png", gap=70)
                    self.rows2 = len(split_flag2)
                    for j, value in enumerate(split_flag2):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Dritte1_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Dritte1_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag2) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag2[j + 1] - value, w=width,
                                     path=self.reg_dir)
            print("split rows Dritte1 Abteilung done")

        if flag == "Dritte2 Abteilung":
            allTestDataName = []
            for filename in os.listdir(self.Dritte2_dir[0:-1]):
                if filename.endswith('.png'):  # 文件名中不包含'pre'字符串
                    # print(filename)
                    allTestDataName.append(filename)

            allTestDataName.sort(key=lambda x: int(x[:-4]))

            for i, image in enumerate(allTestDataName):
                if i < 4:
                    split_flag = rows_cut(self.Dritte2_dir + str(4) + ".png", gap=80)
                    self.rows = len(split_flag)
                    for j, value in enumerate(split_flag):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Dritte2_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Dritte2_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag[j + 1] - value, w=width,
                                     path=self.reg_dir)
                elif i >= 4:
                    split_flag2 = rows_cut(self.Dritte2_dir + str(10) + ".png", gap=70)
                    self.rows2 = len(split_flag2)
                    for j, value in enumerate(split_flag2):
                        # if j >= 6:
                        #     break
                        height, width, img = width_calculate(self.Dritte2_dir + image)
                        # Pixel fine-tuning
                        gray = cv2.imread(self.Dritte2_dir + image, 0)
                        for x in range(0, width):
                            while gray[value, x] != 255:
                                if value == -1:
                                    break
                                value -= 1
                        # cut elements by rows.
                        if j >= len(split_flag2) - 1:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=height - value, w=width, path=self.reg_dir)
                        else:
                            cut_area(img, i=str(i) + str(j), x=0, y=value, h=split_flag2[j + 1] - value, w=width,
                                     path=self.reg_dir)
            print("split rows Dritte2 Abteilung done")

    def recognition(self, flag="Bestandsverzeichnis"):
        excel = excel_operator(path='../data_/38_GBA.xlsx', sheet=flag.replace(" ", "_"))
        if flag == "Bestandsverzeichnis":
            if os.path.isdir(self.reg_dir):
                # iterate rows
                for i in range(self.rows):
                    # iterate columns
                    data = []
                    # same table contain different number of columns?????
                    for j in range(0, 10):
                        try:
                            resize_pixels(image=self.reg_dir + str(j) + str(i) + '.png',
                                          new_path=self.reg_dir + str(j) + str(i) + '.png')
                            data.append(ocr_core(filename=self.reg_dir + str(j) + str(i) + '.png'))
                        except:
                            data.append(" ")

                    excel.write_line(data)
            print("recognize Bestandsverzeichnis done")

        if flag == "Erste Abteilung":
            if os.path.isdir(self.reg_dir):
                # iterate rows
                for i in range(self.rows):
                    # iterate columns
                    data = []
                    for j in range(0, 4):
                        try:
                            resize_pixels(image=self.reg_dir + str(j) + str(i) + '.png',
                                          new_path=self.reg_dir + str(j) + str(i) + '.png')
                            data.append(ocr_core(filename=self.reg_dir + str(j) + str(i) + '.png'))
                        except Exception as e:
                            data.append(" ")
                    for k in range(self.rows2):
                        # iterate columns
                        for w in range(4, 8):
                            try:
                                resize_pixels(image=self.reg_dir + str(w) + str(k) + '.png',
                                              new_path=self.reg_dir + str(w) + str(k) + '.png')
                                data.append(ocr_core(filename=self.reg_dir + str(w) + str(k) + '.png'))
                            except Exception as e:
                                data.append(" ")

                    excel.write_line(data)

            print("recognize Erste Abteilung done")

        if flag == "Zweite Abteilung":
            if os.path.isdir(self.reg_dir):
                # iterate rows
                for i in range(self.rows):
                    # iterate columns
                    data = []
                    for j in range(0, 3):
                        try:
                            resize_pixels(image=self.reg_dir + str(j) + str(i) + '.png',
                                          new_path=self.reg_dir + str(j) + str(i) + '.png')
                            data.append(ocr_core(filename=self.reg_dir + str(j) + str(i) + '.png'))
                        except Exception as e:
                            data.append(" ")
                    if i < self.rows2:
                        for w in range(3, 7):
                            try:
                                resize_pixels(image=self.reg_dir + str(w) + str(i) + '.png',
                                              new_path=self.reg_dir + str(w) + str(i) + '.png')
                                data.append(ocr_core(filename=self.reg_dir + str(w) + str(i) + '.png'))
                            except Exception as e:
                                data.append(" ")

                    excel.write_line(data)
            print("recognize Zweite Abteilung done")

        if flag == "Dritte1 Abteilung":
            if os.path.isdir(self.reg_dir):
                # iterate rows
                for i in range(self.rows):
                    # iterate columns
                    data = []
                    for j in range(0, 4):
                        try:
                            resize_pixels(image=self.reg_dir + str(j) + str(i) + '.png',
                                          new_path=self.reg_dir + str(j) + str(i) + '.png')
                            data.append(ocr_core(filename=self.reg_dir + str(j) + str(i) + '.png'))
                        except Exception as e:
                            data.append(" ")
                    if i < self.rows2:
                        for w in range(4, 10):
                            try:
                                resize_pixels(image=self.reg_dir + str(w) + str(i) + '.png',
                                              new_path=self.reg_dir + str(w) + str(i) + '.png')
                                data.append(ocr_core(filename=self.reg_dir + str(w) + str(i) + '.png'))
                            except Exception as e:
                                data.append(" ")

                    excel.write_line(data)

            print("recognize Dritte1 Abteilung done")

        if flag == "Dritte2 Abteilung":
            if os.path.isdir(self.reg_dir):
                # iterate rows
                for i in range(self.rows):
                    # iterate columns
                    data = []
                    for j in range(0, 4):
                        try:
                            resize_pixels(image=self.reg_dir + str(j) + str(i) + '.png',
                                          new_path=self.reg_dir + str(j) + str(i) + '.png')
                            data.append(ocr_core(filename=self.reg_dir + str(j) + str(i) + '.png'))
                        except Exception as e:
                            data.append(" ")
                    if i < self.rows2:
                        for w in range(4, 10):
                            try:
                                resize_pixels(image=self.reg_dir + str(w) + str(i) + '.png',
                                              new_path=self.reg_dir + str(w) + str(i) + '.png')
                                data.append(ocr_core(filename=self.reg_dir + str(w) + str(i) + '.png'))
                            except Exception as e:
                                data.append(" ")

                    excel.write_line(data)

            print("recognize Dritte2 Abteilung done")


if __name__ == '__main__':
    t = pdf_processing(pdf='../data_/38_GBA/Freudenstadt.pdf')
    # convert pdf to images
    t.pdf_to_images(pdf='../data_/38_GBA/Freudenstadt.pdf')
    t.remove_no_table_images()
    # # cut each columns into images dir, now only test on table of 'Bestandsverzeichnis'
    t.Processing()

    # # # cut rows into pieces, Bestandsverzeichnis, Erste Abteilung, Zweite Abteilung, Dritte1 Abteilung, Dritte2 Abteilung
    # tables = ["Bestandsverzeichnis", "Erste Abteilung", "Zweite Abteilung", "Dritte1 Abteilung"]
    # for i in tables:
    #     t.split_rows(flag=i, gap=40)
    #     t.recognition(flag=i)
    # if t.dritte2:
    #     t.split_rows(flag="Dritte2 Abteilung", gap=40)
    #     t.recognition(flag="Dritte2 Abteilung")

    t.split_rows(flag="Bestandsverzeichnis", gap=40)
    t.recognition(flag="Bestandsverzeichnis")
