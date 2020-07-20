#!/bin/bash

# use CUDNN

#export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cudnn/cudnn-7.5-linux-x64-v5.0/cuda/lib64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/cudnn-7.5-linux-x64-v5.0/cuda/lib64
#export CPATH=$CPATH:/usr/local/cudnn/cudnn-7.5-linux-x64-v5.0/cuda/include

export THEANO_FLAGS=device=cuda,floatX=float32,optimizer_including=cudnn

python -u ./train_nmt.py ./models/
cp -a models models_noise
python -u ./train_nmt_noise.py ./models_noise/



