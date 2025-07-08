#!/bin/bash

#############################################################
### training ###

# for single GPU
# python train.py -opt=options/train/refusion.yml

# for multiple GPUs and specify gpu id in refusion.yml
# torchrun --nproc_per_node=4 --master_port=4321 train.py -opt=options/train/refusion.yml --launcher pytorch

#############################################################

### testing ###
python run_itm.py -opt_path options/test/refusion.yml -input_dir AIM_TEST -output_dir output -pretrained_path lastest_EMA.pth

#############################################################
