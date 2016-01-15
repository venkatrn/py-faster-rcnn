#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn_alt_opt.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn_alt_opt.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

NET_FINAL=output/faster_rcnn_alt_opt/willow_garage_2011_train/ZF_faster_rcnn_final.caffemodel

time ./tools/test_net.py --gpu ${GPU_ID} \
  --def models/${NET}/faster_rcnn_alt_opt/faster_rcnn_test.pt \
  --net ${NET_FINAL} \
  --imdb willow_garage_test \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  ${EXTRA_ARGS}
