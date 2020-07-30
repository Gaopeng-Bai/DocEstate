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
import xlwings as xw


class excel_operator:
    def __init__(self, path="../../data_/38_GBA.xlsx", sheet= "Bestandsverzeichnis"):
        wb = xw.Book(path)
        self.sht = wb.sheets[sheet]

    def write_line(self, value_list, cols=10):
        last_ = self.lastRow(col=cols)
        self.sht.range('A'+str(last_+1)).value = value_list

    def lastRow(self,  col=10):
        """ Find the last row in the worksheet that contains data.

        idx: Specifies the worksheet to select. Starts counting from zero.

        workbook: Specifies the workbook

        col: The column in which to look for the last cell containing data.
        """
        lwr_r_cell = self.sht.cells.last_cell  # lower right cell
        lwr_row = lwr_r_cell.row  # row of the lower right cell
        row = 0
        for i in range(1, col+1):
            lwr_cell = self.sht.range((lwr_row, i))  # change to your specified column
            if lwr_cell.value is None:
                lwr_cell = lwr_cell.end('up')  # go up untill you hit a non-empty cell

            if row < lwr_cell.row:
                row = lwr_cell.row

        return row


if __name__ == '__main__':
    a = excel_operator()
    list = ["1","2","3","4","5","6","7","8","9","10", ]
    a.write_line(list)