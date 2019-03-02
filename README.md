
code for IJCAI 18 paper: Grouping Attribute Recognition for Pedestrian with Joint Recurrent Learning

### runtime environment

- python version : 3.4
- tensorflow version: >= 1.4

### model
    the best model in the paper is in 

### prepare data (use rap as example)
- ROI data: 

use pose estimation model provided in Spindle Net (github link is https://github.com/yokattame/SpindleNet)  to get region proposal data.
    
- put rap or peta label data and region proposal data together.

for format is:

    # 0 
    CAM12_2014-03-05_20140305110334-20140305111754_tarid1199_frame8675_line1.png
    0  0  0  1  0  1  0  1  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    19.75390625 0 68 48.6875
    0 28.796875 68 112.9609375
    0 72.4375 68 188.8125
    49.3738839286 29.6875 68 112.8125
    0 31.3203125 31.921875 97.8203125
    20.4285714286 85.796875 47.5714285714 186.5859375
    8.18247767857 74.8125 35.3253348214 174.5625
    
    that is
    image index
    attribute labels
    head region coordinate
    up region coordinate
    down region coordinate
    left arm coordinate
    right arm coordinate
    left leg coordinate
    eight leg coordinate
- run data/build_rap_region_data.py to get tensorflow TFRecord input


### train

run command for example:

    TRAIN_DIR=DIR for train models
    RAP_DATA_DIR= DIR for rap tensorflow TFRecord file
    MODEL_PATH=DIR for pretrained model
    
    bazel-bin/inception/rap_train \
      --train_dir="${TRAIN_DIR}" \
      --data_dir="${RAP_DATA_DIR}" \
      --pretrained_model_checkpoint_path="${MODEL_PATH}" \
      --fine_tune=False \
      --initial_learning_rate=0.1 \
      --input_queue_memory_factor=1 \
      --num_gpus=1 \
      --max_steps=1001
      
### test
        
    TRAIN_DIR=DIR for train models
    EVAL_DIR= DIR for eval event logs
    RAP_DATA_DIR=DIR for rap tensorflow TFRecord file

        
      bazel-bin/inception/rap_eval \
      --eval_dir="${EVAL_DIR}" \
      --data_dir="${RAP_DATA_DIR}" \
      --subset=validation \
      --num_examples=8317 \
      --checkpoint_dir="${TRAIN_DIR}" \
      --input_queue_memory_factor=1 \
      --run_once
      


