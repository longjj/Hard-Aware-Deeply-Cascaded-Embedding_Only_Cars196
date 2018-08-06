# -*- coding: UTF-8 -*-
import os, sys
import test_cars196_new as test_cars196

def main():
    feature_dim = 128
    query_txt_path = '../feature/score_4832.txt'
    query_label_file = '../data/cars196/test_label.dat'
    data_txt_path = query_txt_path
    data_label_file = query_label_file
    data_cnt = 8131
    top_cnt = 32

    test_cars196.eval(
        query_txt_path,
        query_label_file,
        data_txt_path,
        data_label_file,
        data_cnt,
        feature_dim,
        top_cnt
    )

if __name__=='__main__':
    main()