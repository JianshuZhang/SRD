import sys
import os
import numpy
def cmp_result(rec,label):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def process(recPath, labelPath, resultfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    total_re_dist = 0
    total_re_label = 0
    total_re_line = 0
    total_re_line_rec = 0
    rec_mat = {}
    label_mat = {}
    rec_re_mat = {}
    label_re_mat = {}

    file_list = os.listdir(recPath)
    for file_name in file_list:
        file_key = file_name[:-4]
        recfile = recPath + file_key + '.txt'
        labelfile = labelPath + file_key + '.label'
        rec_mat[file_key] = []
        label_mat[file_key] = []
        rec_re_mat[file_key] = []
        label_re_mat[file_key] = []
        with open(recfile) as f_rec:
            lines = f_rec.readlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split('\t')
                sym = parts[0]
                re = parts[1]
                if line_idx == 0:
                    re = 'Start'
                if line_idx == len(lines)-1:
                    re = 'End'
                rec_mat[file_key] = rec_mat[file_key] + [sym]
                rec_re_mat[file_key] = rec_re_mat[file_key] + [re]

        with open(labelfile) as f_label:
            lines = f_label.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                sym = parts[0]
                re = parts[2]
                label_mat[file_key] = label_mat[file_key] + [sym]
                label_re_mat[file_key] = label_re_mat[file_key] + [re]

    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(rec, label)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    for key_rec in rec_re_mat:
        label_re = label_re_mat[key_rec]
        rec_re = rec_re_mat[key_rec]
        dist_re, llen_re = cmp_result(rec_re, label_re)
        total_re_dist += dist_re
        total_re_label += llen_re
        total_re_line += 1
        if dist_re == 0:
            total_re_line_rec += 1
    cer = float(total_dist)/total_label
    re_cer = float(total_re_dist)/total_re_label

    f_result = open(resultfile,'a')
    f_result.write('CER {}\n'.format(cer))
    f_result.write('reCER {}\n'.format(re_cer))
    f_result.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'compute_sym_re_cer.py recPath labelPath resultfile'
        sys.exit(0)
    process(sys.argv[1], sys.argv[2], sys.argv[3])