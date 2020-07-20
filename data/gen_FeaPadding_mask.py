import sys
import os
import numpy
import cPickle as pkl
import time


def pad_feature_stroke_len_v5():
    root_path = '/home/jszhang/for-demo/data/'
    file_path = root_path + 'file'
    out_file_path = root_path + 'file-pad-v5'
    out_maskfile_path = root_path + 'file-pad-v5-mask'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if not os.path.exists(out_maskfile_path):
        os.mkdir(out_maskfile_path)
    process_num = 0
    pad_points_num = 3
    file_list = os.listdir(file_path)
    for file_name in file_list:
        # file_name = '18_em_1.ascii'
        feature_file = file_path + '/' + file_name
        pad_feature_file = out_file_path + '/' + file_name
        pad_mask_file = out_maskfile_path + '/' + file_name[:-6] + '_mask.txt'
        mat = numpy.loadtxt(feature_file)
        penup_index = numpy.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        p_idx_start = 0
        for pi in range(len(penup_index)):
            stroke_mat = mat[p_idx_start:(penup_index[pi]+1),:]
            if pi < len(penup_index)-1:
                pad_tmp_mat = numpy.zeros((pad_points_num,mat.shape[1]),dtype='float32')
                stroke_mat = numpy.concatenate((stroke_mat,pad_tmp_mat),axis=0)
            if pi == 0:
                pad_mat = stroke_mat
            else:
                pad_mat = numpy.concatenate((pad_mat,stroke_mat),axis=0)
            p_idx_start = penup_index[pi]+1

        pad_mask_mat = numpy.ones([pad_mat.shape[0], 1], dtype='int8')

        for idx in range(len(pad_mat)):
            if pad_mat[idx,-2] == 0. and pad_mat[idx,-1] == 0.:
                pad_mask_mat[idx,:] = 0

        numpy.savetxt(pad_feature_file,pad_mat,fmt='%.6f')
        numpy.savetxt(pad_mask_file,pad_mask_mat,fmt='%d')

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num


def gen_feature_pkl():
    root_path = '/home/jszhang/for-demo/data/'
    feature_path = root_path + 'file-pad-v5/'
    mask_path = root_path + 'file-pad-v5-mask/'
    out_file = root_path + 'test-pad-v5.pkl'
    out_mask_file = root_path + 'test-pad-v5-mask.pkl'
    f_out = open(out_file, 'w')
    f_out_mask = open(out_mask_file, 'w')
    process_num = 0
    features = {}
    masks = {}
    file_list  = os.listdir(feature_path)
    for file_name in file_list:
        key = file_name[:-6]
        feature_file = feature_path + key + '.ascii'
        mat = numpy.loadtxt(feature_file)
        features[key] = mat
        mask_file = mask_path + key + '_mask.txt'
        mmat = numpy.loadtxt(mask_file)
        masks[key] = mmat
        process_num = process_num + 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num

    print 'load ascii file done. files number ', process_num

    pkl.dump(features, f_out)
    pkl.dump(masks, f_out_mask)
    print 'save file done'
    f_out.close()
    f_out_mask.close()


if __name__ == '__main__':

    pad_feature_stroke_len_v5()
    gen_feature_pkl()