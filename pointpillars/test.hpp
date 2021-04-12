#ifndef _TEST_HPP_
#define _TEST_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>
#include "avs_onnx.h"
#include <stdio.h>
#include "post_pointpillars.h"
#include <memory.h>
#include <string.h>

//#include "onnxruntime_c_api.h"
//#include "onnxruntime_cxx_api.h"
using namespace cv;
using namespace std;

void AVS_alloc_mem_tab(AVS_MEM_TAB *mem_tab, int num);

void get_onnx_input(string &input_txt, float *data);

void write_onnx_out(float *pillar_feature, string &pillar_ftxt, int len);

#endif