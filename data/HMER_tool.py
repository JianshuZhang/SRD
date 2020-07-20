#!/usr/bin/env python

import os
import sys
import cPickle as pkl
import numpy
import pdb
import glob
import cv2
from collections import OrderedDict
from math import *
import random
import json_lines
import json
import binascii


def find_brace_structure(in_file = 'caption/train_data_v1.txt',
                         out_file = 'caption/train_structure.txt'):

    structure_list = []
    with open(in_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            for idx in range(len(parts) - 1):
                if parts[idx + 1] == '{':
                    if parts[idx] not in structure_list:
                        structure_list.append(parts[idx])
    f_out = open(out_file, 'w')
    for struc in structure_list:
        f_out.write(struc + '\n')

    f_out.close()


def revise_latex_structure(in_file = 'caption/test_data_v1.txt',
                           out_file = 'caption/test_data_v1_revise.txt'):

# \sqrt [ * ] { * }
# \sqrt { * }
# * _ { * }
# * ^ { * }
# \frac { * } { * }

    f_out = open(out_file, 'w')
    with open(in_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            uid = parts[0]
            tex = parts[1:]
            i = 0
            brace_stack = []
            while i < len(tex):
                if tex[i] == '\sqrt' and tex[i+1] == '[':
                    if tex[i+3] == ']':
                        tex[i+1] = 'sqrt_['
                        tex[i+3] = 'sqrt_]'
                        tex[i+4] = 'sqrt_{'
                        brace_stack.append('sqrt')
                        i += 5
                    else:
                        print 'this caption', uid, '\sqrt [ * ] is not right'
                elif tex[i] == '\sqrt' and tex[i+1] != '[':
                    if tex[i+1] == '{':
                        tex[i+1] = 'sqrt_{'
                        brace_stack.append('sqrt')
                        i += 2
                    else:
                        print 'this caption', uid, '\sqrt { * } is not right'
                elif tex[i] == '_':
                    if tex[i+1] == '{':
                        tex[i+1] = 'sub_{'
                        brace_stack.append('sub')
                        i += 2
                    else:
                        print 'this caption', uid, '_ { * } is not right'
                elif tex[i] == '^':
                    if tex[i+1] == '{':
                        tex[i+1] = 'sup_{'
                        brace_stack.append('sup')
                        i += 2
                    else:
                        print 'this caption', uid, '^ { * } is not right'
                elif tex[i] == '\\frac':
                    if tex[i+1] == '{':
                        tex[i+1] = 'frac_a_{'
                        brace_stack.append('frac_a')
                        i += 2
                    else:
                        print 'this caption', uid, '\\frac { * } is not right'
                elif tex[i] == 'frac_a_}':
                    if tex[i+1] == '{':
                        tex[i+1] = 'frac_b_{'
                        brace_stack.append('frac_b')
                        i += 2
                    else:
                        print 'this caption', uid, '\\frac { * } { * } is not right'
                elif tex[i] == '}':
                    if len(brace_stack) == 0:
                        print 'this caption', uid, 'stack of brace must not Null'
                        print tex
                        print brace_stack
                    if brace_stack[-1] == 'frac_a':
                        tex[i] = brace_stack[-1] + '_}'
                        del brace_stack[-1]
                    else:
                        tex[i] = brace_stack[-1] + '_}'
                        del brace_stack[-1]
                        i += 1
                else:
                    i += 1
            revise_tex = " ".join(tex)
            string = uid + '\t' + revise_tex + '\n'
            f_out.write(string)
    f_out.close()


def norm_id_inkml():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/inkml/CROHME2016/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/data/inkml_norm_id_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    paths = ['TestEM2016']
    process_num = 0
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            inkml_file = root_path + path + '/' + file_name
            norm_inkml_path = out_path + path
            if os.path.exists(norm_inkml_path):
                norm_inkml_file = norm_inkml_path + '/' + file_name
            else:
                os.mkdir(norm_inkml_path)
                norm_inkml_file = norm_inkml_path + '/' + file_name
            #print inkml_file
            with open(inkml_file) as f:
                lines = f.readlines()
                norm_lines = []
                sub_dict = {}
                symbol_stack = []
                mroot_line_list = []
                i = 0
                line_tmp = lines[i].strip()
                while line_tmp != '</annotationXML>':
                    if line_tmp.find('supsub') != -1:
                        print 'warning: this inkml has supsub structure', inkml_file
                        sys.exit()
                    if line_tmp.find('xml:id') == -1:
                        norm_lines.append(lines[i])
                        i += 1
                        line_tmp = lines[i].strip()
                    else:
                        if line_tmp[:6] == '<mfrac':
                            symbol_stack.append('frac')
                            idx = symbol_stack.count('frac')
                            sub_element = line_tmp.split('"')[1]
                            sub_source_str = '"' + sub_element + '"'
                            sub_target_str = '"' + 'frac_' + str(idx) + '"'
                            sub_dict[sub_source_str] = sub_target_str
                            norm_lines.append(lines[i].replace(sub_source_str, sub_target_str))
                        elif line_tmp[:6] == '<msqrt':
                            symbol_stack.append('sqrt')
                            idx = symbol_stack.count('sqrt')
                            sub_element = line_tmp.split('"')[1]
                            sub_source_str = '"' + sub_element + '"'
                            sub_target_str = '"' + 'sqrt_' + str(idx) + '"'
                            sub_dict[sub_source_str] = sub_target_str
                            norm_lines.append(lines[i].replace(sub_source_str, sub_target_str))
                        elif line_tmp[:6] == '<mroot':
                            print 'this inkml has root structure', inkml_file
                            symbol_stack.append('sqrt')
                            idx = symbol_stack.count('sqrt')
                            sub_element = line_tmp.split('"')[1]
                            sub_source_str = '"' + sub_element + '"'
                            sub_target_str = '"' + 'sqrt_' + str(idx) + '"'
                            sub_dict[sub_source_str] = sub_target_str
                            norm_lines.append(lines[i].replace(sub_source_str, sub_target_str))
                            j = i + 1
                            while j < len(lines):
                                root_line_tmp = lines[j].strip()
                                if root_line_tmp != '</mroot>':
                                    j += 1
                                    if root_line_tmp[:6] == '<mroot':
                                        print 'this inkml has loop root structure', inkml_file
                                        break
                                        # sys.exit()
                                else:
                                    mroot_idx = j-1
                                    mroot_line_tmp = lines[mroot_idx].strip()
                                    if mroot_line_tmp[:3] != '<mn' and mroot_line_tmp[:3] != '<mi':
                                        print 'this inkml has special root structure', inkml_file
                                        break
                                    else:
                                        mroot_line_list.append(mroot_idx)
                                        symbol_beg = mroot_line_tmp.find('>') + 1
                                        symbol_end = mroot_line_tmp.rfind('<')
                                        if symbol_end < symbol_beg:
                                            print 'this line', mroot_line_tmp, '>*< wrong'
                                            sys.exit()
                                        symbol = mroot_line_tmp[symbol_beg:symbol_end].strip()
                                        if symbol[0] == '\\':
                                            print 'warning: this inkml has complex root structure', inkml_file
                                            symbol = symbol[1:]
                                        symbol_stack.append(symbol)
                                        idx = symbol_stack.count(symbol)
                                        sub_element = mroot_line_tmp.split('"')[1]
                                        sub_source_str = '"' + sub_element + '"'
                                        sub_target_str = '"' + symbol + '_' + str(idx) + '"'
                                        sub_dict[sub_source_str] = sub_target_str
                                        lines[mroot_idx] = lines[mroot_idx].replace(sub_source_str, sub_target_str)
                                        break
                            if j == len(lines):
                                print 'this inkml has wrong root structure', inkml_file
                                sys.exit()
                        else:
                            if i in mroot_line_list:
                                norm_lines.append(lines[i])
                            else:
                                symbol_beg = line_tmp.find('>') + 1
                                symbol_end = line_tmp.rfind('<')
                                if symbol_end < symbol_beg:
                                    print 'this line', line_tmp, '>*< wrong'
                                    sys.exit()
                                symbol = line_tmp[symbol_beg:symbol_end].strip()
                                if symbol[0] == '\\':
                                    symbol = symbol[1:]
                                symbol_stack.append(symbol)
                                idx = symbol_stack.count(symbol)
                                sub_element = line_tmp.split('"')[1]
                                sub_source_str = '"' + sub_element + '"'
                                sub_target_str = '"' + symbol + '_' + str(idx) + '"'
                                sub_dict[sub_source_str] = sub_target_str
                                norm_lines.append(lines[i].replace(sub_source_str, sub_target_str))
                        i += 1
                        line_tmp = lines[i].strip()
                while i < len(lines):
                    line_tmp = lines[i].strip()       
                    if line_tmp[:19] != '<annotationXML href':
                        norm_lines.append(lines[i])
                    else:
                        line_sub_tmp = lines[i]
                        for sub_str in sub_dict:
                            if line_sub_tmp.find(sub_str) != -1:
                                line_sub_tmp = line_sub_tmp.replace(sub_str, sub_dict[sub_str])
                                break
                        norm_lines.append(line_sub_tmp)
                    i += 1

            if len(lines) != len(norm_lines):
                print 'this inkml', inkml_file, 'processing error'
                sys.exit()
            f_out = open(norm_inkml_file,'w')
            for norm_line in norm_lines:
                f_out.write(norm_line)

            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num



def inkml2lg():
    crohmelib_bin_path = '/lustre1/hw/jszhang6/git/crohmelib/bin/'
    inkml_path = '/lustre1/hw/jszhang6/HMER/srd/data/inkml_norm_id_r1/CROHME2016/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_id_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    paths = ['TestEM2016']
    process_num = 0
    for path in paths:
        file_list  = os.listdir(inkml_path + path)
        for file_name in file_list:
            inkml_file = inkml_path + path + '/' + file_name
            lg_path = out_path + path
            if os.path.exists(lg_path):
                lg_file = lg_path + '/' + file_name[:-6] + '.lg'
            else:
                os.mkdir(lg_path)
                lg_file = lg_path + '/' + file_name[:-6] + '.lg'

            order = 'perl ' + crohmelib_bin_path + 'crohme2lg.pl -s ' + inkml_file + ' ' + lg_file
            try:
                os.system(order)
            except:
                print 'this file', inkml_file, 'inkml2lg pl error'
                # sys.exit()
            else:
                process_num += 1

            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num


def check_LGdict():
    latex_dictionary = 'dictionary.txt'
    latex_dict = []
    with open(latex_dictionary) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                latex_dict.append(parts[0])

    f_out = open('check_dict2.txt','w')
    LG_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v2_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    paths = ['TestEM2016']
    process_num = 0
    error_dict = []
    for path in paths:
        file_list  = os.listdir(LG_path + path)
        for file_name in file_list:
            lg_file = LG_path + path + '/' + file_name
            with open(lg_file) as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(', ')
                    if parts[0] == 'O':
                        if parts[3] != '1.0':
                            print 'this file', lg_file, '1.0 format error', line
                            sys.exit()
                        sym_pos = parts[1]
                        sym_pos_tmp = sym_pos.split('_')
                        if len(sym_pos_tmp) != 2:
                            print 'this file', lg_file, 'sym_pos format error', line
                            sys.exit()
                        sym = parts[2]
                        if len(sym) == 1 and sym != sym_pos_tmp[0]:
                            error_str = sym + ' to1 ' + sym_pos_tmp[0]
                            if error_str not in error_dict:
                                error_dict.append(error_str)
                                error_dict.append(lg_file)
                        if len(sym) > 1 and sym != 'COMMA' and sym[1:] != sym_pos_tmp[0]:
                            error_str = sym + ' to2 ' + sym_pos_tmp[0]
                            if error_str not in error_dict:
                                error_dict.append(error_str)
                                error_dict.append(lg_file)
                        if sym not in latex_dict:
                            error_str = sym + ' to3 ' + sym_pos_tmp[0]
                            if error_str not in error_dict:
                                error_dict.append(error_str)
                                error_dict.append(lg_file)
                        if len(sym) > 1 and sym != 'COMMA' and sym[0] != '\\':
                            error_str = sym + ' to4 ' + sym_pos_tmp[0]
                            if error_str not in error_dict:
                                error_dict.append(error_str)
                                error_dict.append(lg_file)
                        if sym == 'COMMA' and sym_pos_tmp[0] != 'COMMA':
                            error_str = sym + ' to5 ' + sym_pos_tmp[0]
                            if error_str not in error_dict:
                                error_dict.append(error_str)
                                error_dict.append(lg_file)
                        if sym == ',':
                            print 'this file', lg_file, 'has unexpected COMMA', line
                            sys.exit()
            process_num += 1

    if process_num / 1000 == process_num * 1.0 / 1000:
        print 'process files', process_num

    for error_str in error_dict:
        f_out.write(error_str + '\n')
    f_out.close()


def norm_lg_v2():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_id_r1/CROHME2016/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v2_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    paths = ['TestEM2016']
    process_num = 0
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            lg_file = root_path + path + '/' + file_name
            norm_lg_path = out_path + path
            if os.path.exists(norm_lg_path):
                norm_lg_file = norm_lg_path + '/' + file_name
            else:
                os.mkdir(norm_lg_path)
                norm_lg_file = norm_lg_path + '/' + file_name
            with open(lg_file) as f:
                lines = f.readlines()
                norm_lines = []
                sub_dict = {}
                AUTO_num = 0
                for line in lines:
                    parts = line.strip().split(', ')
                    if parts[0] == 'O':
                        sym_pos = parts[1]
                        sym_pos_tmp = sym_pos.split('_')
                        sym = parts[2]
                        if sym_pos_tmp[0] == 'AUTO':
                            AUTO_num += 1
                        elif sym == '-' and sym_pos_tmp[0] == 'frac':
                            parts[2] = '\\frac'
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '-' and sym_pos_tmp[0] == '=':
                            parts[2] = '='
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '-' and sym_pos_tmp[0] == '=':
                            parts[2] = '='
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\lt':
                            parts[2] = '<'
                            sym_pos_tmp[0] = '<'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            if sym_pos != sym_pos_sub:
                                for test_line in lines:
                                    if test_line.find(sym_pos_sub) != -1:
                                        print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                        sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\gt':
                            parts[2] = '>'
                            sym_pos_tmp[0] = '>'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            if sym_pos != sym_pos_sub:
                                for test_line in lines:
                                    if test_line.find(sym_pos_sub) != -1:
                                        print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                        sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\ldots' and sym_pos_tmp[0] == 'ctdot':
                            sym_pos_tmp[0] = 'ldots'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\neq' and sym_pos_tmp[0] == 'ne':
                            sym_pos_tmp[0] = 'neq'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\lim' and sym_pos_tmp[0] == 'im':
                            sym_pos_tmp[0] = 'lim'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\infty' and sym_pos_tmp[0] == 'infin':
                            sym_pos_tmp[0] = 'infty'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\rightarrow' and sym_pos_tmp[0] == 'rarr':
                            sym_pos_tmp[0] = 'rightarrow'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\leq' and sym_pos_tmp[0] == 'le':
                            sym_pos_tmp[0] = 'leq'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\geq' and sym_pos_tmp[0] == 'ge':
                            sym_pos_tmp[0] = 'geq'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\exists' and sym_pos_tmp[0] == 'exist':
                            sym_pos_tmp[0] = 'exists'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        elif sym == '\\ldots' and sym_pos_tmp[0] == 'hellip':
                            sym_pos_tmp[0] = 'ldots'
                            sym_pos_sub = '_'.join(sym_pos_tmp)
                            for test_line in lines:
                                if test_line.find(sym_pos_sub) != -1:
                                    print 'lg file norm error', lg_file, 'sub occupied', sym_pos_sub
                                    sys.exit()
                            sub_dict[sym_pos] = sym_pos_sub
                            parts[1] = sym_pos_sub
                            norm_lines.append(', '.join(parts) + '\n')
                        else:
                            norm_lines.append(line)
                    elif parts[0] == 'EO':
                    # elif parts[0] == 'R':
                        if len(sub_dict) > 0:
                            line_sub_tmp = line
                            for sub_str in sub_dict:
                                # if line_sub_tmp.find(sub_str) != -1:
                                line_sub_tmp = line_sub_tmp.replace(sub_str, sub_dict[sub_str])
                                    # break
                            norm_lines.append(line_sub_tmp)
                        else:
                            norm_lines.append(line)
                    else:
                        norm_lines.append(line)

            if len(lines) - AUTO_num != len(norm_lines):
                print 'this lg', lg_file, 'processing error'
                sys.exit()
            f_out = open(norm_lg_file,'w')
            for norm_line in norm_lines:
                f_out.write(norm_line)
            f_out.close()
            #sys.exit()

            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num

                
def check_node_freq():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v3_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    paths = ['TestEM2016']
    process_num = 0
    f_out = open('check_node_freq.txt','w')
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            lg_file = root_path + path + '/' + file_name
            with open(lg_file) as f:
                lines = f.readlines()
                sym_stack = []
                nodes_stack = []
                for line in lines:
                    line = line.strip()
                    parts = line.split(', ')
                    if parts[0] == 'O':
                        if parts[1] not in sym_stack:
                            sym_stack.append(parts[1])
                    if parts[0] == 'EO':
                    # if parts[0] == 'R':
                        if parts[2] in nodes_stack:
                            print 'this lg file', lg_file, 'nodes more than once'
                            f_out.write(line + '\n')
                            f_out.write(lg_file + '\n')
                            #sys.exit()
                        elif parts[2] not in sym_stack:
                            print 'this lg file', lg_file, 'has informal nodes'
                            print parts[2]
                            # sys.exit()
                        else:
                            nodes_stack.append(parts[2])

            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num

    f_out.close()


def norm_node_freq():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v2_r1/CROHME2016/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    paths = ['TestEM2016']
    out_root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v3_r1/CROHME2016/'
    process_num = 0
    for path in paths:
        # out_path = out_root_path + path
        out_path = out_root_path
        if not os.path.exists(out_path):
            os.mkdir(out_path) 
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            # file_name = '27_em_118.lg'
            lg_file = root_path + path + '/' + file_name
            norm_lg_file = out_path + '/' + file_name
            f_out = open(norm_lg_file, 'w')
            EO_lines = []
            rm_lines = []
            SRT_dict = {}
            with open(lg_file) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    parts = line.split(', ')
                    if parts[0] != 'EO':
                    # if parts[0] != 'R':
                        f_out.write(line+'\n')
                        if parts[0] == 'O':
                            SRT_dict[parts[1]] = []
                    else:
                        EO_lines.append(line)
                for idx, line in enumerate(EO_lines):
                    parts = line.split(', ')
                    if parts[1] not in SRT_dict or parts[2] not in SRT_dict:
                        print 'this lg file', lg_file, 'has informal nodes'
                        sys.exit()
                    SRT_dict[parts[2]].append([parts[1],parts[3],idx])
                for item in SRT_dict:
                    if len(SRT_dict[item])==2:
                        if SRT_dict[item][0][1] != 'Inside' and SRT_dict[item][1][1] != 'Inside':
                            print 'this lg file', lg_file, 'has other than inside problems'
                            sys.exit()
                        elif SRT_dict[item][0][1] == 'Inside' and SRT_dict[item][1][1] != 'Inside':
                            if SRT_dict[SRT_dict[item][1][0]][0][1] == 'Inside' and len(SRT_dict[SRT_dict[item][1][0]]) == 1:
                                rm_lines.append(SRT_dict[item][0][2])
                            else:
                                print 'this lg file', lg_file, 'has wrong multi syms'
                                sys.exit()
                        elif SRT_dict[item][0][1] != 'Inside' and SRT_dict[item][1][1] == 'Inside':
                            if SRT_dict[SRT_dict[item][0][0]][0][1] == 'Inside' and len(SRT_dict[SRT_dict[item][0][0]]) == 1:
                                rm_lines.append(SRT_dict[item][1][2])
                            else:
                                print 'this lg file', lg_file, 'has wrong multi syms'
                                sys.exit()
                        else:
                            print 'this lg file', lg_file, 'has double inside problems'
                            sys.exit()
                    if len(SRT_dict[item])>2:
                        print 'this lg file', lg_file, 'has 2 more related syms'
                        sys.exit()
            for idx, line in enumerate(EO_lines):
                if idx not in rm_lines:
                    f_out.write(line+'\n')
            f_out.close()
            # sys.exit()
            if len(rm_lines) > 0:
                print 'this lg file', lg_file, 'has been normed'

            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num



def get_srd_label():
    # RIT_2014_19 need manually revised
    latex_caption_file = '/lustre1/hw/jszhang6/HMER/srd/data/caption/train_data_v1.txt'
    latex_caption = {}
    with open(latex_caption_file) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('\t')
            symbol_stack = []
            if len(parts) == 2:
                key = parts[0]
                caption = parts[1]
                caption = caption.replace(',', 'COMMA')
                # caption = caption.replace('{', '')
                # caption = caption.replace('}', '') # will destroy \{ \}
                caption = caption.replace('\\limits', '')
                caption = caption.replace('\cdots', '\ldots') # ctdot question
                caption = caption.replace('\cdot', '.') # ctdot question
                caption = caption.strip().split()
                id_caption = []
                for sym in caption:
                    if sym == '{' or sym == '}':
                        continue
                    elif sym == '_' or sym == '^':
                        id_caption.append(sym)
                    elif sym == 'COMMA':
                        symbol_stack.append('COMMA')
                        idx = symbol_stack.count('COMMA')
                        sub_symbol = 'COMMA_' + str(idx)
                        id_caption.append(sub_symbol)
                    elif len(sym) > 1:
                        if sym[0] != '\\':
                            print 'this file', key, 'caption', parts[1], 'has informal symbol', sym
                            sys.exit()
                        else:
                            sym = sym[1:] # remove \
                            symbol_stack.append(sym)
                            idx = symbol_stack.count(sym)
                            sub_symbol = sym + '_' + str(idx)
                            id_caption.append(sub_symbol)
                    elif len(sym) == 1:
                        symbol_stack.append(sym)
                        idx = symbol_stack.count(sym)
                        sub_symbol = sym + '_' + str(idx)
                        id_caption.append(sub_symbol)
                    else:
                        print 'this file', key, 'caption', parts[1], 'has unknown symbol', sym
                        sys.exit()

                latex_caption[key] = id_caption

    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG_norm_v2_r1/CROHME2014/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd_r1/CROHME2014/'
    paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    # paths = ['TestEM2014']
    # paths = ['MathBrush']
    process_num = 0
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            lg_file = root_path + path + '/' + file_name
            key = file_name[:-3] # remove .lg
            srd_path = out_path + path
            if os.path.exists(srd_path):
                srd_file = srd_path + '/' + key + '.srd'
            else:
                os.mkdir(srd_path)
                srd_file = srd_path + '/' + key + '.srd'
            
            sym_segment = {}
            sym_relation = {}
            with open(lg_file) as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split(', ', 4)
                    if parts[0] == 'O':
                        if parts[3] != '1.0':
                            print 'this lg file Objects is informal', lg_file
                            sys.exit()
                        if parts[1] not in sym_segment:
                            sym_segment[parts[1]] = parts[2] + '\t' + parts[4]
                        else:
                            print 'this lg file has occupied sym_pos', lg_file
                            sys.exit()

                    if parts[0] == 'EO':
                    # if parts[0] == 'R':
                        if parts[4] != '1.0':
                            print 'this lg file SRT is informal', lg_file
                            sys.exit()
                        if parts[2] not in sym_relation:
                            sym_relation[parts[2]] = []
                            sym_relation[parts[2]].append([parts[1],parts[3]])
                        else:
                            sym_relation[parts[2]].append([parts[1],parts[3]])
            # print key
            f_out = open(srd_file,'w')
            id_caption = latex_caption[key]
            if path == 'MathBrush':
                if 'ldots_1' in id_caption and 'ldots_1' not in sym_segment and '._1' in sym_segment:
                    new_symbol_stack = []
                    new_caption = []
                    new_id_caption = []
                    for id_cap in id_caption:
                        if id_cap == '_' or id_cap == '^':
                            new_caption.append(id_cap)
                        else:
                            new_caption.append(id_cap.split('_')[0])
                    new_caption_str = ' '.join(new_caption)
                    new_caption_str = new_caption_str.replace('ldots','. . .')
                    new_caption = new_caption_str.split(' ')
                    for sym in new_caption:
                        if sym == '_' or sym == '^':
                            new_id_caption.append(sym)
                        else:
                            new_symbol_stack.append(sym)
                            idx = new_symbol_stack.count(sym)
                            sub_symbol = sym + '_' + str(idx)
                            new_id_caption.append(sub_symbol)

                    id_caption = new_id_caption

            out_str = sym_segment[id_caption[0]] + '\t<s>\t-1\tStart' + '\n'
            f_out.write(out_str)
            if len(id_caption) > 1:
                for sym in id_caption[1:]:
                    if sym == '_' or sym == '^':
                        continue
                    elif sym not in sym_segment:
                        if sym.split('_')[0] == '[' or sym.split('_')[0] == ']':
                            continue
                        print 'this lg file has unknown objects', lg_file
                        print ' '.join(id_caption)
                        print sym
                        sys.exit()
                    elif sym not in sym_relation:
                        if sym.split('_')[0] == '[' or sym.split('_')[0] == ']':
                            continue
                        print 'this lg file has unknown SRT', lg_file
                        print ' '.join(id_caption)
                        print sym
                        sys.exit()
                    else:
                        out_str = sym_segment[sym]
                        if len(sym_relation[sym]) == 1:
                            out_str += '\t' + sym_segment[sym_relation[sym][0][0]] + '\t' + sym_relation[sym][0][1]
                            if sym_relation[sym][0][1] == 'Above':
                                previous_sym = id_caption[id_caption.index(sym) - 1]
                                relation_sym = sym_relation[sym][0][0].split('_')[0]
                                if previous_sym != '^' and relation_sym != 'frac' and relation_sym != 'sqrt':
                                    print 'sup relation is wrong', lg_file
                                    sys.exit()
                            if sym_relation[sym][0][1] == 'Below':
                                previous_sym = id_caption[id_caption.index(sym) - 1]
                                relation_sym = sym_relation[sym][0][0].split('_')[0]
                                if previous_sym != '_' and relation_sym != 'frac':
                                    print 'sub relation is wrong', lg_file
                                    sys.exit()
                        else:
                            # print 'this lg file has multiple node relation', lg_file
                            distance_list = []
                            for i in range(len(sym_relation[sym])):
                                distance = id_caption.index(sym) - id_caption.index(sym_relation[sym][i][0])
                                if distance <= 0:
                                    print 'relation is not forward', lg_file
                                    sys.exit()
                                else:
                                    distance_list.append(distance)
                            min_index = distance_list.index(min(distance_list))
                            out_str += '\t' + sym_segment[sym_relation[sym][min_index][0]] + '\t' + sym_relation[sym][min_index][1]
                            if sym_relation[sym][min_index][1] == 'Above':
                                previous_sym = id_caption[id_caption.index(sym) - 1]
                                relation_sym = sym_relation[sym][min_index][0].split('_')[0]
                                if previous_sym != '^' and relation_sym != 'frac' and relation_sym != 'sqrt':
                                    print 'sup relation is wrong', lg_file
                                    sys.exit()
                            if sym_relation[sym][min_index][1] == 'Below':
                                previous_sym = id_caption[id_caption.index(sym) - 1]
                                relation_sym = sym_relation[sym][min_index][0].split('_')[0]
                                if previous_sym != '_' and relation_sym != 'frac':
                                    print 'sub relation is wrong', lg_file
                                    sys.exit()
                    f_out.write(out_str + '\n')
                out_str = '</s>\t-1\t' + sym_segment[id_caption[-1]] + '\tEnd' + '\n'
                f_out.write(out_str)
            else:
                out_str = '</s>\t-1\t' + sym_segment[id_caption[-1]] + '\tEnd' + '\n'
                f_out.write(out_str)
            f_out.close()

            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num



def gen_feature_pkl():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd/CROHME2014/'
    paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    #paths = ['TestEM2014']
    feature_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/train-dis-0.005-revise-pad-v2/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    out_file = out_path + '8feature-train-dis-0.005-revise-pad-v2.pkl'
    f_out = open(out_file, 'w')
    process_num = 0
    features = {}
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .srd
            feature_file = feature_path + key + '.ascii'
            mat = numpy.loadtxt(feature_file)
            features[key] = mat
            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'load ascii file done. files number ', process_num

    pkl.dump(features, f_out)
    print 'save file done'
    f_out.close()

def gen_feature_pkl_v3():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/LG/'
    # paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    paths = ['TestEM2016_r1_n1']
    feature_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/16-test-dis-0.005-revise-pad-v5/'
    mask_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/16-test-dis-0.005-revise-pad-v5-mask/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    out_file = out_path + '16-9feature-test-dis-0.005-revise-pad-v5.pkl'
    out_mask_file = out_path + '16-9feature-test-dis-0.005-revise-pad-v5-mask.pkl'
    f_out = open(out_file, 'w')
    f_out_mask = open(out_mask_file, 'w')
    process_num = 0
    features = {}
    masks = {}
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            # key = file_name[:-4] # remove suffix .srd
            key = file_name[:-3] # remove suffix .lg
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


def gen_valid_pkl():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    test_pkl_file = root_path + '8feature-test-dis-0.005-revise-pad-v2.pkl'
    f_in = open(test_pkl_file)
    all_features = pkl.load(f_in)

    valid_list = root_path + 'valid_data_v3.txt'
    valid_pkl_file = root_path + '8feature-valid-dis-0.005-revise-pad-v2.pkl'
    f_out = open(valid_pkl_file, 'w')
    features = {}
    with open(valid_list) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            key = parts[0]
            if key not in all_features:
                print 'warning: this file not in test.pkl', key
            else:
                features[key] = all_features[key]

    pkl.dump(features, f_out)
    print 'save file done'
    f_in.close()
    f_out.close()

def gen_valid_pkl_v3():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    test_pkl_file = root_path + '9feature-test-dis-0.005-revise-pad-v5.pkl'
    test_mask_file = root_path + '9feature-test-dis-0.005-revise-pad-v5-mask.pkl'
    f_in = open(test_pkl_file)
    f_in_mask = open(test_mask_file)
    all_features = pkl.load(f_in)
    all_masks = pkl.load(f_in_mask)

    valid_list = root_path + 'valid_data_v3.txt'
    valid_pkl_file = root_path + '9feature-valid-dis-0.005-revise-pad-v5.pkl'
    valid_mask_file = root_path + '9feature-valid-dis-0.005-revise-pad-v5-mask.pkl'
    f_out = open(valid_pkl_file, 'w')
    f_out_mask = open(valid_mask_file, 'w')
    features = {}
    masks = {}
    with open(valid_list) as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            key = parts[0]
            if key not in all_features:
                print 'warning: this file not in test.pkl', key
            else:
                features[key] = all_features[key]
                masks[key] = all_masks[key]

    pkl.dump(features, f_out)
    pkl.dump(masks, f_out_mask)
    print 'save file done'
    f_in.close()
    f_out.close()
    f_in_mask.close()
    f_out_mask.close()



def gen_srd_label():
    mode = 'test'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd/CROHME2014/'
    if mode == 'train':
        paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    elif mode == 'test':
        paths = ['TestEM2014']
    else:
        print 'unknown run mode'
        sys.xit()
    
    out_label_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/label/' + mode + '/'
    out_root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    if mode == 'train':
        out_file_align = out_root_path + 'align-train-dis-0.005-revise-pad-v3.pkl'
        out_file_related_align = out_root_path + 'related-align-train-dis-0.005-revise-pad-v3.pkl'
        feature_file = out_root_path + '9feature-train-dis-0.005-revise-pad-v3.pkl'
        outpkl_label_file = out_root_path + 'train-label.pkl'
    else:
        out_file_align = out_root_path + 'align-test-dis-0.005-revise-pad-v3.pkl'
        out_file_related_align = out_root_path + 'related-align-test-dis-0.005-revise-pad-v3.pkl'
        feature_file = out_root_path + '9feature-test-dis-0.005-revise-pad-v3.pkl'
        outpkl_label_file = out_root_path + 'test-label.pkl'

    feature_fp = open(feature_file)
    features = pkl.load(feature_fp)

    f_out_align = open(out_file_align, 'w')
    f_out_related_align = open(out_file_related_align, 'w')
    out_label_fp = open(outpkl_label_file, 'w')
    alignment = {}
    related_alignment = {}
    label_lines = {}
    process_num = 0
    
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .srd
            if os.path.exists(out_label_path):
                out_label_file = out_label_path + '/' + key + '.label'
            else:
                os.mkdir(out_label_path)
                out_label_file = out_label_path + '/' + key + '.label'
            f_out_label = open(out_label_file, 'w')
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = 0
                align_list = []
                realign_list = []
                label_strs = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        wordNum += 1
                        sym = parts[0]
                        align_list.append(parts[1])
                        related_sym = parts[2]
                        realign_list.append(parts[3])
                        relation = parts[4]
                        string = sym + '\t' + related_sym + '\t' + relation
                        label_strs.append(string)
                        f_out_label.write(string + '\n')
                    else:
                        print 'illegal line', key
                        sys.exit()
                f_out_label.close()
                label_lines[key] = label_strs

                fea = features[key]
                align = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                realign = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                penup_index = numpy.where(fea[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
                wordNum = -1
                for align_str in align_list:
                    wordNum += 1
                    align_str_parts = align_str.split(', ')
                    for i in range(len(align_str_parts)):
                        pos = int(align_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            align[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            align[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                wordNum = -1
                for realign_str in realign_list:
                    wordNum += 1
                    realign_str_parts = realign_str.split(', ')
                    for i in range(len(realign_str_parts)):
                        pos = int(realign_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            realign[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            realign[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                alignment[key] = align
                related_alignment[key] = realign

            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'process files number ', process_num

    pkl.dump(alignment, f_out_align)
    pkl.dump(related_alignment, f_out_related_align)
    pkl.dump(label_lines, out_label_fp)
    print 'save file done'
    f_out_align.close()
    f_out_related_align.close()
    out_label_fp.close()


def gen_srd_label_v3():
    mode = 'train'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd/CROHME2014/'
    if mode == 'train':
        paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    elif mode == 'test':
        paths = ['TestEM2014']
    else:
        print 'unknown run mode'
        sys.xit()
    
    out_label_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/label/' + mode + '/'
    out_root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    if mode == 'train':
        out_file_align = out_root_path + 'align-train-dis-0.005-revise-pad-v4.pkl'
        out_file_related_align = out_root_path + 'related-align-train-dis-0.005-revise-pad-v4.pkl'
        feature_file = out_root_path + '9feature-train-dis-0.005-revise-pad-v4.pkl'
        mask_file = out_root_path + '9feature-train-dis-0.005-revise-pad-v4-mask.pkl'
        outpkl_label_file = out_root_path + 'train-label.pkl'
    else:
        out_file_align = out_root_path + 'align-test-dis-0.005-revise-pad-v4.pkl'
        out_file_related_align = out_root_path + 'related-align-test-dis-0.005-revise-pad-v4.pkl'
        feature_file = out_root_path + '9feature-test-dis-0.005-revise-pad-v4.pkl'
        mask_file = out_root_path + '9feature-test-dis-0.005-revise-pad-v4-mask.pkl'
        outpkl_label_file = out_root_path + 'test-label.pkl'

    feature_fp = open(feature_file)
    features = pkl.load(feature_fp)
    mask_fp = open(mask_file)
    masks = pkl.load(mask_fp)

    f_out_align = open(out_file_align, 'w')
    f_out_related_align = open(out_file_related_align, 'w')
    out_label_fp = open(outpkl_label_file, 'w')
    alignment = {}
    related_alignment = {}
    label_lines = {}
    process_num = 0
    
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .srd
            if os.path.exists(out_label_path):
                out_label_file = out_label_path + '/' + key + '.label'
            else:
                os.mkdir(out_label_path)
                out_label_file = out_label_path + '/' + key + '.label'
            f_out_label = open(out_label_file, 'w')
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = 0
                align_list = []
                realign_list = []
                label_strs = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        wordNum += 1
                        sym = parts[0]
                        align_list.append(parts[1])
                        related_sym = parts[2]
                        realign_list.append(parts[3])
                        relation = parts[4]
                        string = sym + '\t' + related_sym + '\t' + relation
                        label_strs.append(string)
                        f_out_label.write(string + '\n')
                    else:
                        print 'illegal line', key
                        sys.exit()
                f_out_label.close()
                label_lines[key] = label_strs

                fea = features[key]
                mask = masks[key]
                align = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                realign = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                penup_index = numpy.where(fea[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
                pp_start = 0
                for pi in range(len(penup_index)):
                    half_pad_num = len(mask[pp_start:(penup_index[pi]+1)]) - mask[pp_start:(penup_index[pi]+1)].sum()
                    penup_index[pi] += half_pad_num
                    pp_start = penup_index[pi] + 1
                # if len(mask) != mask.sum():
                #     print key
                #     print penup_index
                    # sys.exit()

                wordNum = -1
                for align_str in align_list:
                    wordNum += 1
                    align_str_parts = align_str.split(', ')
                    for i in range(len(align_str_parts)):
                        pos = int(align_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            align[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            align[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                wordNum = -1
                for realign_str in realign_list:
                    wordNum += 1
                    realign_str_parts = realign_str.split(', ')
                    for i in range(len(realign_str_parts)):
                        pos = int(realign_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            realign[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            realign[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                alignment[key] = align
                related_alignment[key] = realign

            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'process files number ', process_num

    pkl.dump(alignment, f_out_align)
    pkl.dump(related_alignment, f_out_related_align)
    pkl.dump(label_lines, out_label_fp)
    print 'save file done'
    f_out_align.close()
    f_out_related_align.close()
    out_label_fp.close()



def gen_voc():
    label_path = '/home/jszhang/HMER/srd/prepare_data/data/label/train/'
    out_path = '/home/jszhang/HMER/srd/prepare_data/data/'

    f_out_char_voc = open(out_path + 'dictionary.txt', 'w')
    f_out_relation_voc = open(out_path + 'relation_dictionary.txt', 'w')

    dictionary = []
    relation_dictionary = []
    file_list = os.listdir(label_path)
    for file_name in file_list:
        label_file = label_path + file_name
        with open(label_file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split('\t')
                sym = parts[0]
                relation = parts[2]
                if sym not in dictionary:
                    dictionary.append(sym)
                if relation not in relation_dictionary:
                    relation_dictionary.append(relation)

    dictionary.remove('</s>')
    for i in range(len(dictionary)):
        f_out_char_voc.write('{}\t{}\n'.format(dictionary[i],i+1))
    f_out_char_voc.write('</s>\t0\n')
    f_out_char_voc.close()

    relation_dictionary.remove('Start')
    relation_dictionary.remove('End')
    for i in range(len(relation_dictionary)):
        f_out_relation_voc.write('{}\t{}\n'.format(relation_dictionary[i],i+2))
    f_out_relation_voc.write('Start\t0\n')
    f_out_relation_voc.write('End\t1\n')
    f_out_relation_voc.close()



def check_feature_stroke_len():
    mode = 'test'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    if mode == 'train':
        file_path = root_path + 'train-dis-0.005-revise'
    else:
        file_path = root_path + 'test-dis-0.005-revise'
    process_num = 0
    file_list  = os.listdir(file_path)
    for file_name in file_list:
        feature_file = file_path + '/' + file_name
        mat = numpy.loadtxt(feature_file)
        penup_index = numpy.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        pointNum = 0
        idx_start = 0
        for idx in penup_index:
            if idx - idx_start < 5:
                print file_name
                print idx
                sys.exit()
            idx_start = idx

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num


def pad_feature_stroke_len_v2():
    mode = 'train'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    if mode == 'train':
        file_path = root_path + 'train-dis-0.005-revise'
        out_file_path = root_path + 'train-dis-0.005-revise-pad-v2'
    else:
        file_path = root_path + 'test-dis-0.005-revise'
        out_file_path = root_path + 'test-dis-0.005-revise-pad-v2'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    process_num = 0
    pad_points_num = 6
    file_list = os.listdir(file_path)
    for file_name in file_list:
        feature_file = file_path + '/' + file_name
        pad_feature_file = out_file_path + '/' + file_name
        mat = numpy.loadtxt(feature_file)
        penup_index = numpy.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        pad_dict = {}
        pad_lines_num = 0
        pp_idx_start = -1
        for pp_idx in penup_index:
            stroke_points_num = pp_idx - pp_idx_start
            if stroke_points_num < pad_points_num:
                pad_dict[pp_idx_start+1] = pad_points_num - stroke_points_num
                pad_lines_num += pad_points_num - stroke_points_num
            pp_idx_start = pp_idx

        pad_mat_shape = [len(mat)+pad_lines_num, 8]
        pad_mat = numpy.zeros([pad_mat_shape[0], pad_mat_shape[1]], dtype='float32')

        idx_pad = 0
        for idx in range(len(mat)):
            if idx not in pad_dict:
                pad_mat[idx_pad,:-2] = mat[idx,:-3]
                pad_mat[idx_pad,-2:] = mat[idx,-2:]
                idx_pad += 1
            else:
                pad_mat[idx_pad:idx_pad+pad_dict[idx],:2] = mat[idx,:2]
                pad_mat[idx_pad:idx_pad+pad_dict[idx],-2] = 1.
                pad_mat[idx_pad:idx_pad+pad_dict[idx],-1] = 0.
                pad_mat[idx_pad-1,4] = pad_mat[idx_pad-1,2]
                pad_mat[idx_pad-1,5] = pad_mat[idx_pad-1,3]
                if idx+1 < len(mat):
                    pad_mat[idx_pad+pad_dict[idx]-1,-4] = mat[idx+1,0] - mat[idx,0]
                    pad_mat[idx_pad+pad_dict[idx]-1,-3] = mat[idx+1,1] - mat[idx,1]
                else:
                    pad_mat[idx_pad+pad_dict[idx]-1,-4] = 0.
                    pad_mat[idx_pad+pad_dict[idx]-1,-3] = 0.
                idx_pad = idx_pad + pad_dict[idx]
                pad_mat[idx_pad,:-2] = mat[idx,:-3]
                pad_mat[idx_pad,-2:] = mat[idx,-2:]
                idx_pad += 1
        numpy.savetxt(pad_feature_file,pad_mat,fmt='%.6f')
        if len(pad_dict) > 0:
            print file_name
            #print pad_dict
            #sys.exit()

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num



def pad_feature_stroke_len_v3():
    mode = 'test'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    if mode == 'train':
        file_path = root_path + 'train-dis-0.005-revise'
        out_file_path = root_path + 'train-dis-0.005-revise-pad-v3'
        out_maskfile_path = root_path + 'train-dis-0.005-revise-pad-v3-mask'
    else:
        file_path = root_path + 'test-dis-0.005-revise'
        out_file_path = root_path + 'test-dis-0.005-revise-pad-v3'
        out_maskfile_path = root_path + 'test-dis-0.005-revise-pad-v3-mask'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if not os.path.exists(out_maskfile_path):
        os.mkdir(out_maskfile_path)
    process_num = 0
    pad_points_num = 7
    file_list = os.listdir(file_path)
    for file_name in file_list:
        # file_name = '518_em_438.ascii'
        feature_file = file_path + '/' + file_name
        pad_feature_file = out_file_path + '/' + file_name
        pad_mask_file = out_maskfile_path + '/' + file_name[:-6] + '_mask.txt'
        mat = numpy.loadtxt(feature_file)
        penup_index = numpy.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        pad_dict = {}
        pad_lines_num = 0
        pp_idx_start = -1
        for pp_idx in penup_index:
            stroke_points_num = pp_idx - pp_idx_start
            if stroke_points_num < pad_points_num:
                for pad_numi in range(1, (pad_points_num-1)/2+1):
                    if 2*pad_numi + stroke_points_num >= pad_points_num:
                        if pp_idx_start+1 in pad_dict:
                            pad_dict[pp_idx_start+1] += pad_numi
                        else:
                            pad_dict[pp_idx_start+1] = pad_numi
                        if pp_idx_start+1+stroke_points_num in pad_dict:
                            pad_dict[pp_idx_start+1+stroke_points_num] += pad_numi
                        else:
                            pad_dict[pp_idx_start+1+stroke_points_num] = pad_numi
                        pad_lines_num += 2*pad_numi
                        break
            pp_idx_start = pp_idx

        pad_mat_shape = [mat.shape[0]+pad_lines_num, mat.shape[1]]
        pad_mat = numpy.zeros([pad_mat_shape[0], pad_mat_shape[1]], dtype='float32')
        pad_mask_mat = numpy.ones([pad_mat_shape[0], 1], dtype='int8')

        idx_pad = 0
        for idx in range(len(mat)):
            if idx not in pad_dict:
                pad_mat[idx_pad,:] = mat[idx,:]
                idx_pad += 1
            else:
                pad_mask_mat[idx_pad:idx_pad+pad_dict[idx]] = 0
                idx_pad = idx_pad + pad_dict[idx]
                pad_mat[idx_pad,:] = mat[idx,:]
                idx_pad += 1
        numpy.savetxt(pad_feature_file,pad_mat,fmt='%.6f')
        numpy.savetxt(pad_mask_file,pad_mask_mat,fmt='%d')
        if len(pad_dict) > 0:
            print file_name
            # print pad_dict
            # sys.exit()

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num


def pad_feature_stroke_len_v4():
    mode = 'train'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    if mode == 'train':
        file_path = root_path + 'train-dis-0.005-revise'
        out_file_path = root_path + 'train-dis-0.005-revise-pad-v4'
        out_maskfile_path = root_path + 'train-dis-0.005-revise-pad-v4-mask'
    else:
        file_path = root_path + 'test-dis-0.005-revise'
        out_file_path = root_path + 'test-dis-0.005-revise-pad-v4'
        out_maskfile_path = root_path + 'test-dis-0.005-revise-pad-v4-mask'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if not os.path.exists(out_maskfile_path):
        os.mkdir(out_maskfile_path)
    process_num = 0
    pad_points_num = 9
    file_list = os.listdir(file_path)
    for file_name in file_list:
        # file_name = '518_em_438.ascii'
        feature_file = file_path + '/' + file_name
        pad_feature_file = out_file_path + '/' + file_name
        pad_mask_file = out_maskfile_path + '/' + file_name[:-6] + '_mask.txt'
        mat = numpy.loadtxt(feature_file)
        penup_index = numpy.where(mat[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        p_idx_start = 0
        pp_idx_start = -1
        for pi in range(len(penup_index)):
            stroke_mat = mat[p_idx_start:(penup_index[pi]+1),:]
            stroke_points_num = penup_index[pi] - pp_idx_start
            if stroke_points_num < pad_points_num:
                for pad_numi in range(1, (pad_points_num-1)/2+1):
                    if 2*pad_numi + stroke_points_num >= pad_points_num:
                        pad_tmp_mat = numpy.zeros((pad_numi,mat.shape[1]),dtype='float32')
                        stroke_mat = numpy.concatenate((pad_tmp_mat,stroke_mat),axis=0)
                        stroke_mat = numpy.concatenate((stroke_mat,pad_tmp_mat),axis=0)
                        break
            if pi == 0:
                pad_mat = stroke_mat
            else:
                pad_mat = numpy.concatenate((pad_mat,stroke_mat),axis=0)
            pp_idx_start = penup_index[pi]
            p_idx_start = penup_index[pi]+1

        pad_dict = {}
        pad_lines_num = 0
        pp_idx_start = -1
        for pp_idx in penup_index:
            stroke_points_num = pp_idx - pp_idx_start
            if stroke_points_num < pad_points_num:
                for pad_numi in range(1, (pad_points_num-1)/2+1):
                    if 2*pad_numi + stroke_points_num >= pad_points_num:
                        if pp_idx_start+1 in pad_dict:
                            pad_dict[pp_idx_start+1] += pad_numi
                        else:
                            pad_dict[pp_idx_start+1] = pad_numi
                        if pp_idx_start+1+stroke_points_num in pad_dict:
                            pad_dict[pp_idx_start+1+stroke_points_num] += pad_numi
                        else:
                            pad_dict[pp_idx_start+1+stroke_points_num] = pad_numi
                        pad_lines_num += 2*pad_numi
                        break
            pp_idx_start = pp_idx

        pad_mask_mat = numpy.ones([pad_mat.shape[0], 1], dtype='int8')

        for idx in range(len(pad_mat)):
            if pad_mat[idx,-2] == 0. and pad_mat[idx,-1] == 0.:
                pad_mask_mat[idx,:] = 0

        numpy.savetxt(pad_feature_file,pad_mat,fmt='%.6f')
        numpy.savetxt(pad_mask_file,pad_mask_mat,fmt='%d')
        if len(pad_dict) > 0:
            print file_name
            # print pad_dict
            # sys.exit()

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num


# pad 3 zero-vectors between every 2 strokes
def pad_feature_stroke_len_v5():
    mode = 'train'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    if mode == 'train':
        file_path = root_path + 'train-dis-0.005-revise'
        out_file_path = root_path + 'train-dis-0.005-revise-pad-v5'
        out_maskfile_path = root_path + 'train-dis-0.005-revise-pad-v5-mask'
    else:
        file_path = root_path + 'test-dis-0.005-revise'
        out_file_path = root_path + 'test-dis-0.005-revise-pad-v5'
        out_maskfile_path = root_path + 'test-dis-0.005-revise-pad-v5-mask'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if not os.path.exists(out_maskfile_path):
        os.mkdir(out_maskfile_path)
    process_num = 0
    pad_points_num = 3
    file_list = os.listdir(file_path)
    for file_name in file_list:
        # file_name = '518_em_438.ascii'
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

# pad 3 zero-vectors between every 2 strokes
def pad_16pklfeature_stroke_len_v5():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/'
    pklfile_path = root_path + '16-9feature-test-dis-0.005-revise.pkl'
    f_pkl = open(pklfile_path)
    features16 = pkl.load(f_pkl)
    f_pkl.close()
    out_file_path = root_path + '16-test-dis-0.005-revise-pad-v5'
    out_maskfile_path = root_path + '16-test-dis-0.005-revise-pad-v5-mask'
    if not os.path.exists(out_file_path):
        os.mkdir(out_file_path)
    if not os.path.exists(out_maskfile_path):
        os.mkdir(out_maskfile_path)
    process_num = 0
    pad_points_num = 3
    # file_list = os.listdir(file_path)
    for file_name in features16:
        # file_name = '518_em_438.ascii'
        # feature_file = file_path + '/' + file_name
        pad_feature_file = out_file_path + '/' + file_name + '.ascii'
        pad_mask_file = out_maskfile_path + '/' + file_name + '_mask.txt'
        mat = features16[file_name]
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



def gen_srd_label_v5():
    mode = 'test'
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/srd_r1/CROHME2014/'
    if mode == 'train':
        paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    elif mode == 'test':
        paths = ['TestEM2014']
    else:
        print 'unknown run mode'
        sys.xit()
    
    out_label_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/label_r1/' + mode + '/'
    out_root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'
    if mode == 'train':
        out_file_align = out_root_path + 'align-train-dis-0.005-revise-pad-v5-r1.pkl'
        out_file_related_align = out_root_path + 'related-align-train-dis-0.005-revise-pad-v5-r1.pkl'
        feature_file = out_root_path + '9feature-train-dis-0.005-revise-pad-v5.pkl'
        mask_file = out_root_path + '9feature-train-dis-0.005-revise-pad-v5-mask.pkl'
        outpkl_label_file = out_root_path + 'train-label-r1.pkl'
    else:
        out_file_align = out_root_path + 'align-test-dis-0.005-revise-pad-v5-r1.pkl'
        out_file_related_align = out_root_path + 'related-align-test-dis-0.005-revise-pad-v5-r1.pkl'
        feature_file = out_root_path + '9feature-test-dis-0.005-revise-pad-v5.pkl'
        mask_file = out_root_path + '9feature-test-dis-0.005-revise-pad-v5-mask.pkl'
        outpkl_label_file = out_root_path + 'test-label-r1.pkl'

    feature_fp = open(feature_file)
    features = pkl.load(feature_fp)
    mask_fp = open(mask_file)
    masks = pkl.load(mask_fp)

    f_out_align = open(out_file_align, 'w')
    f_out_related_align = open(out_file_related_align, 'w')
    out_label_fp = open(outpkl_label_file, 'w')
    alignment = {}
    related_alignment = {}
    label_lines = {}
    process_num = 0
    
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .srd
            if os.path.exists(out_label_path):
                out_label_file = out_label_path + '/' + key + '.label'
            else:
                os.mkdir(out_label_path)
                out_label_file = out_label_path + '/' + key + '.label'
            f_out_label = open(out_label_file, 'w')
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = 0
                align_list = []
                realign_list = []
                label_strs = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        wordNum += 1
                        sym = parts[0]
                        align_list.append(parts[1])
                        related_sym = parts[2]
                        realign_list.append(parts[3])
                        relation = parts[4]
                        string = sym + '\t' + related_sym + '\t' + relation
                        label_strs.append(string)
                        f_out_label.write(string + '\n')
                    else:
                        print 'illegal line', key
                        sys.exit()
                f_out_label.close()
                label_lines[key] = label_strs

                fea = features[key]
                mask = masks[key]
                align = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                realign = numpy.zeros([fea.shape[0], wordNum], dtype='int8')
                penup_index = numpy.where(fea[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
                pp_start = 0
                for pi in range(len(penup_index)-1):
                    half_pad_num = 3
                    penup_index[pi] += half_pad_num
                    pp_start = penup_index[pi] + 1
                # if len(mask) != mask.sum():
                #     print key
                #     print penup_index
                #     sys.exit()

                wordNum = -1
                for align_str in align_list:
                    wordNum += 1
                    align_str_parts = align_str.split(', ')
                    for i in range(len(align_str_parts)):
                        pos = int(align_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            align[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            align[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                wordNum = -1
                for realign_str in realign_list:
                    wordNum += 1
                    realign_str_parts = realign_str.split(', ')
                    for i in range(len(realign_str_parts)):
                        pos = int(realign_str_parts[i])
                        if pos == -1:
                            continue
                        elif pos == 0:
                            realign[0:(penup_index[pos]+1), wordNum] = 1
                        else:
                            realign[(penup_index[pos-1]+1):(penup_index[pos]+1), wordNum] = 1

                alignment[key] = align
                related_alignment[key] = realign

            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'process files number ', process_num

    pkl.dump(alignment, f_out_align)
    pkl.dump(related_alignment, f_out_related_align)
    pkl.dump(label_lines, out_label_fp)
    print 'save file done'
    f_out_align.close()
    f_out_related_align.close()
    out_label_fp.close()



def draw_online_images():
    root_path = '/lustre1/hw/jszhang6/HMER/srd/data/features/test-dis-0.005-revise/'
    out_path = '/lustre1/hw/jszhang6/HMER/srd/data/images/test/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    file_list  = os.listdir(root_path)
    process_num = 0
    for file_name in file_list:
        # file_name = '18_em_0.ascii'
        key = file_name[:-6] # remove suffix .ascii
        out_file = out_path + key + '.bmp'
        # out_file = 'test.bmp'
        feature_file = root_path + file_name
        fea = numpy.loadtxt(feature_file)
        x = 200*fea[:,0]
        y = 200*fea[:,1]
        width = int(numpy.round(x.max()-x.min())+40)
        height = int(numpy.round(y.max()-y.min())+40)
        img = 255*numpy.ones([height, width], dtype='uint8')
        penup_index = numpy.where(fea[:,-1] == 1)[0] # 0 denote pen down, 1 denote pen up
        img_x = numpy.round(x-x.min())+20
        img_y = numpy.round(y-y.min())+20
        pp_start = 0
        for stroke in penup_index:
            for p_idx in range(pp_start,stroke):
                point_begin = (int(img_x[p_idx]), int(img_y[p_idx]))
                point_end = (int(img_x[p_idx+1]), int(img_y[p_idx+1]))
                cv2.line(img,point_begin,point_end,0,5,cv2.CV_AA)
            pp_start = stroke + 1
        # img = numpy.tile(img,[3,1,1])
        # img[2] = 255
        # img = img.transpose(1,2,0)
        cv2.imwrite(out_file,img)
        # sys.exit()

        process_num += 1
        if process_num / 500 == process_num * 1.0 / 500:
            print 'process files', process_num


def srd2gtd():
    
    root_path = '/bfs1/cv1/jszhang6/HMER/srd/data/srd_r1/CROHME2014/'
    out_root_path = '/bfs1/cv1/jszhang6/HMER/srd/data/gtd/CROHME2014/'
    paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB','TestEM2014']
    process_num = 0
    for path in paths:
        out_path = out_root_path + path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .srd
            out_gtd_file = out_path + '/' + key + '.gtd'
            f_out = open(out_gtd_file, 'w')
            sub_dict = {}
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = 1
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        sym = parts[0]
                        align = parts[1]
                        sub_dict[align] = str(wordNum)
                    else:
                        print 'illegal line', key
                        sys.exit()
                    wordNum += 1
                for line_id in range(len(lines)):
                    line = lines[line_id]
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        sym = parts[0]
                        align = parts[1]
                        sym_pos = sub_dict[align]
                        related_sym = parts[2]
                        realign = parts[3]
                        resym_pos = sub_dict[realign]
                        if line_id == 0:
                            resym_pos = str(0)
                        relation = parts[4]
                    else:
                        print 'illegal line', key
                        sys.exit()
                    out_str = sym + '\t' + sym_pos + '\t' + related_sym + '\t' + resym_pos + '\t' + relation + '\n'
                    f_out.write(out_str)
                f_out.close()
            process_num += 1
            if process_num / 1000 == process_num * 1.0 / 1000:
                print 'process files', process_num


def gen_gtd_label():
    mode = 'train'
    root_path = '/bfs1/cv1/jszhang6/HMER/srd/data/gtd/CROHME2014/'
    if mode == 'train':
        paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    elif mode == 'test':
        paths = ['TestEM2014']
    else:
        print 'unknown run mode'
        sys.xit()

    out_root_path = '/bfs1/cv1/jszhang6/HMER/srd/prepare_data/data/'
    if mode == 'train':
        outpkl_label_file = out_root_path + 'train-label-gtd-v1.pkl'
    else:
        outpkl_label_file = out_root_path + 'test-label-gtd-v1.pkl'

    out_label_fp = open(outpkl_label_file, 'w')
    label_lines = {}
    process_num = 0
    
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .gtd
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                label_strs = []
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        sym = parts[0]
                        align = parts[1]
                        related_sym = parts[2]
                        realign = parts[3]
                        relation = parts[4]
                        string = sym + '\t' + align + '\t' + related_sym + '\t' + realign + '\t' + relation
                        label_strs.append(string)
                    else:
                        print 'illegal line', key
                        sys.exit()
                label_lines[key] = label_strs

            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'process files number ', process_num

    pkl.dump(label_lines, out_label_fp)
    print 'save file done'
    out_label_fp.close()


def gen_gtd_align():
    mode = 'train'
    root_path = '/bfs1/cv1/jszhang6/HMER/srd/data/gtd/CROHME2014/'
    if mode == 'train':
        paths = ['expressmatch','extension','HAMEX','KAIST','MathBrush','MfrDB']
    elif mode == 'test':
        paths = ['TestEM2014']
    else:
        print 'unknown run mode'
        sys.xit()

    out_root_path = '/bfs1/cv1/jszhang6/HMER/srd/prepare_data/data/'
    if mode == 'train':
        outpkl_label_file = out_root_path + 'train-label-align-gtd-v1.pkl'
    else:
        outpkl_label_file = out_root_path + 'test-label-align-gtd-v1.pkl'

    out_label_fp = open(outpkl_label_file, 'w')
    label_aligns = {}
    process_num = 0
    
    for path in paths:
        file_list  = os.listdir(root_path + path)
        for file_name in file_list:
            key = file_name[:-4] # remove suffix .gtd
            with open(root_path + path + '/' + file_name) as f:
                lines = f.readlines()
                wordNum = len(lines)
                align = numpy.zeros([wordNum, wordNum], dtype='int8')
                wordindex = -1
                for line in lines:
                    wordindex += 1
                    parts = line.strip().split('\t')
                    if len(parts) == 5:
                        realign = parts[3]
                        realign_index = int(realign)
                        align[realign_index,wordindex] = 1
                    else:
                        print 'illegal line', key
                        sys.exit()
                label_aligns[key] = align

            process_num = process_num + 1
            if process_num / 500 == process_num * 1.0 / 500:
                print 'process files', process_num

    print 'process files number ', process_num

    pkl.dump(label_aligns, out_label_fp)
    print 'save file done'
    out_label_fp.close()

if __name__ == '__main__':

    # find_brace_structure()
    # revise_latex_structure()
    # norm_id_inkml()
    # inkml2lg()
    # check_LGdict()
    # norm_lg_v2()
    # norm_node_freq()
    # check_node_freq()
    # get_srd_label()
    # gen_feature_pkl()
    # gen_valid_pkl()
    # gen_srd_label()
    # gen_srd_label_v3()
    # gen_srd_label_v5()
    # gen_voc()
    # check_feature_stroke_len()
    # pad_feature_stroke_len_v2()
    # pad_feature_stroke_len_v4()
    # pad_feature_stroke_len_v5()
    # pad_16pklfeature_stroke_len_v5()
    # gen_feature_pkl_v3()
    # gen_valid_pkl_v3()
    # draw_online_images()
    # srd2gtd()
    # gen_gtd_label()
    gen_gtd_align()