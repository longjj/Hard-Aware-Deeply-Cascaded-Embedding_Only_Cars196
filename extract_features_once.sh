CAFFEROOT=/export/home/v-jianjie/code/caffe
TOOLS=$CAFFEROOT/build/tools
$TOOLS/extract_features2file model/HDC/HDC_CARS196/HDC_solver_iter_2000.caffemodel HDC_deploy.prototxt loss3/classifier_norm feature/score_2000 8131 txt GPU 0