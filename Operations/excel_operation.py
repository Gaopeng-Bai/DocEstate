#!/usr/bin/python3
# -*-coding:utf-8 -*-

# Reference:**********************************************
# @Time    : 7/23/2020 9:43 AM
# @Author  : Gaopeng.Bai
# @File    : excel_operation.py
# @User    : gaope
# @Software: PyCharm
# @Description: 
# Reference:**********************************************

import xlrd as xw
import xlwt as xt
from xlutils.copy import copy


class excel_operator:
    def __init__(self, path="../../data_/38_GBA.xlsx", sheet="Bestandsverzeichnis"):
        data = xw.open_workbook(path)
        self.read_sht = data.sheet_by_name(sheet)

        ws = copy(data)
        self.write_sht = ws.get_sheet(sheet)
        rows = self.read_sht.nrows

        self.write_sht.write(rows+1, 0, "haha")
        self.write_sht.write(rows+1, 3, "haha")
        # self.write_sht.write(5, 3, "haha")
        ws.save("../../data_/38_GBA.xls")
        print(self.write_sht.nrows)

    def write_line(self, value_list):
        last_ = self.write_sht.nrows
        columns = self.write_sht.ncols
        for i in range(columns):
            self.write_sht.write(last_ + 1, i, value_list[i])

    def lastRow(self,  col=10):
        """ Find the last row in the worksheet that contains data.

        idx: Specifies the worksheet to select. Starts counting from zero.

        workbook: Specifies the workbook

        col: The column in which to look for the last cell containing data.
        """
        lwr_r_cell = self.write_sht.cells.last_cell  # lower right cell
        lwr_row = lwr_r_cell.row  # row of the lower right cell
        row = 0
        for i in range(1, col+1):
            lwr_cell = self.write_sht.range((lwr_row, i))  # change to your specified column
            if lwr_cell.value is None:
                lwr_cell = lwr_cell.end('up')  # go up untill you hit a non-empty cell

            if row < lwr_cell.row:
                row = lwr_cell.row

        return row


if __name__ == '__main__':
    a = excel_operator()
    list = ["1","2","3","4","5","6","7","8","9","10", ]
    a.write_line(list)