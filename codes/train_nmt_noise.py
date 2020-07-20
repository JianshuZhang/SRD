import numpy
import os
import sys

from nmt_noise import train

root_path = '/lustre1/hw/jszhang6/HMER/srd/prepare_data/data/'

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     dim_relation=params['dim_relation'][0],
                     dim_enc=params['dim_enc'], # multi layer
                     dim_dec=params['dim_dec'][0], 
                     dim_coverage=params['dim_coverage'][0], 
                     down_sample=params['down_sample'],
                     dim_attention=params['dim_attention'][0],
                     dim_reattention=params['dim_reattention'][0],
                     dim_target=params['dim_target'][0],
                     dim_retarget=params['dim_retarget'][0],
                     dim_feature=params['dim_feature'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     model_cost_coeff=params['model_cost_coeff'][0],
                     la=params['lambda-align'][0],
                     lb=params['lambda-realign'][0],
                     optimizer=params['optimizer'][0], 
                     patience=12,
                     max_xlen=params['max_xlen'][0],
                     max_ylen=params['max_ylen'][0],
                     batch_size=8,
                     valid_batch_size=8,
                     validFreq=-1,
                     validStart=-10,
                     dispFreq=100,
                     saveFreq=-1,
                     sampleFreq=-1,
          datasets=[root_path + '9feature-train-dis-0.005-revise-pad-v5.pkl',
                    root_path + '9feature-train-dis-0.005-revise-pad-v5-mask.pkl',
                    root_path + 'train-label-r1.pkl',
                    root_path + 'align-train-dis-0.005-revise-pad-v5-r1.pkl',
                    root_path + 'related-align-train-dis-0.005-revise-pad-v5-r1.pkl'],
          valid_datasets=[root_path + '9feature-valid-dis-0.005-revise-pad-v5.pkl',
                    root_path + '9feature-valid-dis-0.005-revise-pad-v5-mask.pkl',
                    root_path + 'test-label-r1.pkl',
                    root_path + 'align-test-dis-0.005-revise-pad-v5-r1.pkl',
                    root_path + 'related-align-test-dis-0.005-revise-pad-v5-r1.pkl'],
          dictionaries=[root_path + 'dictionary.txt',
                        root_path + '6relation_dictionary.txt',],
          valid_output=['./result/symbol_relation/',
                        './result/alignment/',
                        './result/relation_alignment/'],
          valid_result=['./result/valid.cer'],
          use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    
    max_xlen=[2000]
    max_ylen=[400]
    modelDir=sys.argv[1]
    dim_word=[256]
    dim_relation=[256]
    dim_dec=[256]
    dim_coverage=[121]
    dim_enc=[256,256,256,256] # they are bidirectional
    down_sample=[0,0,1,1]
    dim_attention=[512]
    dim_reattention=[512]

        
    main(0, {
        'model': [modelDir+'attention_max_xlen'+str(max_xlen)+'_dimWord'+str(dim_word[0])+'_dim'+str(dim_dec[0])+'.npz'],
        'dim_word': dim_word,
        'dim_relation': dim_relation,
        'dim_dec': dim_dec,
        'dim_coverage': dim_coverage,
        'dim_enc': dim_enc,
        'down_sample':down_sample,
        'dim_attention': dim_attention,
        'dim_reattention': dim_reattention,
        'dim_target': [103], 
        'dim_retarget': [6], 
        'dim_feature': [9], 
        'optimizer': ['adadelta_weightnoise'],
        'decay-c': [0.], 
        'clip-c': [1000.], 
        'use-dropout': [False],
        'model_cost_coeff':[0.1],
        'learning-rate': [1e-8],
        'lambda-align': [0.1],
        'lambda-realign': [0.1],
        'max_xlen': max_xlen,
        'max_ylen': max_ylen,
        'reload': [True]})
