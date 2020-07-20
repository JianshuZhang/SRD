import sys
import os
import numpy
import cPickle as pkl
import time

def cmp_result(out_lines,label_lines):
    rec = True

    if len(out_lines) != len(label_lines):
        rec = False
    else:
        for idx in range(len(label_lines)):
            out_line = out_lines[idx]
            label_line = label_lines[idx]
            out_parts = out_line.split('\t')
            label_parts = label_line.split('\t')

            out_sym = out_parts[0]
            label_sym = label_parts[0]
            out_resym = out_parts[1]
            label_resym = label_parts[1]
            out_re = out_parts[2]
            label_re = label_parts[2]
            out_resym_s = out_resym.split('_')[0]
            label_resym_s = label_resym.split('_')[0]

            if (out_resym_s == '\lim' and label_resym_s == '\lim') or \
            (out_resym_s == '\int' and label_resym_s == '\int') or \
            (out_resym_s == '\sum' and label_resym_s == '\sum'):
                if out_re == 'Above':
                    out_re = 'Sup'
                if out_re == 'Below':
                    out_re = 'Sub'
                if label_re == 'Above':
                    label_re = 'Sup'
                if label_re == 'Below':
                    label_re = 'Sub'

            if out_sym != label_sym or out_resym != label_resym or out_re != label_re:
                rec = False
                break
    return rec


def process_rec(symreFile,aliFile,realiFile):
    sym_idx_list = []
    resym_idx_list = []
    resym_idx_list.append('<s>')
    re_list = []
    symbol_stack = []
    out_lines = []
    with open(symreFile) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            sym = parts[0]
            re = parts[1]
            symbol_stack.append(sym)
            sidx = symbol_stack.count(sym)
            sym_idx = sym + '_' + str(sidx)
            sym_idx_list.append(sym_idx)
            re_list.append(re)
    re_list[0] = 'Start'
    re_list[-1] = 'End'
    ali = numpy.loadtxt(aliFile)
    reali = numpy.loadtxt(realiFile)
    seqLen = reali.shape[0]
    for idx in range(1, seqLen-1):
        sym_reali = reali[idx,:]
        dist_list = []
        for j in range(idx):
            cmp_ali = ali[j,:]
            dist = numpy.sum(numpy.square(cmp_ali-sym_reali))
            dist_list.append(dist)
        ali_idx = dist_list.index(min(dist_list))
        resym_idx_list.append(sym_idx_list[ali_idx])
    for i in range(seqLen-1):
        out_line = sym_idx_list[i] + '\t' + resym_idx_list[i] + '\t' + re_list[i]
        out_lines.append(out_line)

    return out_lines


def process_srd(srdFile):
    with open(srdFile) as f:
        ali_sym_dict = {}
        symbol_stack = []
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            sym = parts[0]
            ali = parts[1]
            symbol_stack.append(sym)
            sidx = symbol_stack.count(sym)
            sym_idx = sym + '_' + str(sidx)
            ali_sym_dict[ali] = sym_idx
        srd_lines = []
        symbol_stack = []
        for line_idx, line in enumerate(lines[:-1]):
        # the last line </s> is ignored
            parts = line.strip().split('\t')
            sym = parts[0]
            reali = parts[3]
            re = parts[4]
            symbol_stack.append(sym)
            sidx = symbol_stack.count(sym)
            sym_idx = sym + '_' + str(sidx)
            if line_idx == 0:
            # first line
                srd_line = sym_idx + '\t<s>\tStart'
            else:
                srd_line = sym_idx + '\t' + ali_sym_dict[reali] + '\t' + re
            srd_lines.append(srd_line)
    return srd_lines

def process(recPath):
    label_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd_r1/CROHME2014/TestEM2014/'
    rec_sym_re_path = recPath + 'symbol_relation/'
    rec_ali_path = recPath + 'alignment/'
    rec_reali_path = recPath + 'relation_alignment/'
    out_result_file = recPath + 'test_ExpRate_result.txt'

    f_out = open(out_result_file,'w')

    file_list  = os.listdir(rec_sym_re_path)
    total_num = len(file_list)
    print 'total numbers', total_num
    correct_num = 0
    for file_name in file_list:
        # file_name = '18_em_1.txt'
        key = file_name[:-4]
        # mask = masks[key]
        out_file = rec_sym_re_path + file_name
        out_ali_file = rec_ali_path + key + '_align.txt'
        out_reali_file = rec_reali_path + key + '_realign.txt'
        out_lines = process_rec(out_file,out_ali_file,out_reali_file)
        label_file = label_path + key + '.srd'
        label_lines = process_srd(label_file)

        rec_result = cmp_result(out_lines,label_lines)
        if rec_result:
            correct_num += 1
    correct_rate = 100. * correct_num / total_num
    f_out.write('ExpRate %.2f' % (correct_rate))
    f_out.close()        


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'evaluate_ExpRate.py recPath'
        sys.exit(0)
    process(sys.argv[1])