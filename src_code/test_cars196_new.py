# -*- coding: UTF-8 -*-
from util import *
import numpy as np
from time import sleep
# from eval_metric import *

def get_recall(dists, test_label_vector, data_label_vector, top_count):
    num_query = dists.shape[0]
    correct_radio = np.zeros((6, num_query))
    for i in xrange(num_query):
        # each i represent one query
        labels_sorted = np.array(data_label_vector)[np.argsort(dists[i, :])].flatten()
        labels = labels_sorted[0:top_count]
        correct_count = np.zeros((6, 1))
        for j in xrange(top_count):
            if labels[j] == test_label_vector[i]:
                if (j < 1):
                    correct_count[0] = correct_count[0] + 1
                if (j < 2):
                    correct_count[1] = correct_count[1] + 1
                if (j < 4):
                    correct_count[2] = correct_count[2] + 1
                if (j < 8):
                    correct_count[3] = correct_count[3] + 1
                if (j < 16):
                    correct_count[4] = correct_count[4] + 1
                if (j < 32):
                    correct_count[5] = correct_count[5] + 1
        correct_radio[0][i] = (correct_count[0] > 0)
        correct_radio[1][i] = (correct_count[1] > 0)
        correct_radio[2][i] = (correct_count[2] > 0)
        correct_radio[3][i] = (correct_count[3] > 0)
        correct_radio[4][i] = (correct_count[4] > 0)
        correct_radio[5][i] = (correct_count[5] > 0)
    ave_precision = np.mean(correct_radio, axis=1)

    print 'stanford cars mean recall@ 1 : %f' % (ave_precision[0])
    print 'stanford cars mean recall@ 2 : %f' % (ave_precision[1])
    print 'stanford cars mean recall@ 4 : %f' % (ave_precision[2])
    sleep(0.2)
    print 'stanford cars mean recall@ 8 : %f' % (ave_precision[3])
    print 'stanford cars mean recall@ 16 : %f' % (ave_precision[4])
    print 'stanford cars mean recall@ 32 : %f' % (ave_precision[5])

def compute_distances_self(set_list):
    """
    通过 python 的序列化操作加速计算两两之间的距离
    Args:
        set_list (n x k np.array): n 个 k 维的数据 
    """
    print 'start compute distances ...'
    num_set = set_list.shape[0]
    dists = np.zeros((num_set, num_set))

    M = np.dot(set_list, set_list.T)
    te = np.sum(set_list ** 2, axis=1)
    tr = np.sum(set_list ** 2, axis=1)
    dists = np.sqrt(np.abs(-2 * M + tr + np.matrix(te).T))
    print 'finished compute distances ...'
    return dists

def load_feature_txt(path, count):
    res = np.loadtxt(path)
    return res[:count, :]

def eval(query_txt_path, query_label_file, data_txt_path, data_label_file, data_cnt, feature_dim, top_cnt):
    query_label_vector = pickle.load(open(query_label_file, 'rb'))
    data_label_vector = pickle.load(open(data_label_file, 'rb'))
    data_feature_vector = load_feature_txt(data_txt_path, data_cnt)
    # dists 是一个矩阵，每个值表示两张图片的距离值
    # 在这里的距离还是最为 normal 的 L2 距离
    dists = compute_distances_self(data_feature_vector)
    # 把自己和自己的距离给抹去
    np.fill_diagonal(dists, float("inf"))
    # 只需要两两之间的距离矩阵 dists
    # 以及 query 和 data 的 labels
    # 就可以查找每一张 query 对应的前 top_cnt 的匹配
    get_recall(dists, query_label_vector, data_label_vector, top_cnt)