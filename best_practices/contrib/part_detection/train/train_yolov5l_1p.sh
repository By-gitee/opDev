#!/bin/bash

# wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt

# Train
nohup python3 train.py --weights ./yolov5l.pt \
                        --cfg ./models/myyolov5l.yaml \
                        --data ./datasets/mycoco1.yaml \
                        --device 0 \
                        --batch-size 32 \
                        --workers 4 \
                        --epochs 300 \
                        --checkpoint yolov5l_mycoco1.pt > result_yolov5l_train.txt 2>&1 &

# Val
nohup python3 val.py --data ./datasets/mycoco1.yaml \
                    --weights ./yolov5l_mycoco1.pt \
                    --device 0 \
                    --batch-size 32 \
                    --conf-thres 0.01 \
                    --iou-thres 0.5 > result_yolov5l_val.txt 2>&1 &