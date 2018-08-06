CAFFEROOT=/export/home/v-jianjie/code/caffe
TOOLS=$CAFFEROOT/build/tools
STANDARDNET=/export/home/v-jianjie/net/standardnet
BASENET=bvlc_googlenet

GLOG_logtostderr=0 GLOG_log_dir=log/ \
$TOOLS/caffe train --solver=HDC_solver.prototxt --gpu 0 --weights=$STANDARDNET/$BASENET/bvlc_googlenet.caffemodel

