#ifndef _AVS_ONNX_LIB_H_
#define _AVS_ONNX_LIB_H_

#include "avs_onnx.h"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <string>
#include <vector>
#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime_session_options_config_keys.h"
#include "cuda_provider_factory.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

typedef struct _AVS_ONNX_FILTER_ {
	// dnnģ�����ñ���
	Mat				image;		// ͼ��
	dnn::Net		net;		// opencv dnn
	// onnxruntime preģ�����ñ���
	Ort::Env		env;		// one enviroment per process, ��Ҫ���ֱ�������Ч
	Ort::Session	session;	// onnxruntime
	// onnxruntime rpnģ�����ñ���
	Ort::Env		rpn_env;		// one enviroment per process, ��Ҫ���ֱ�������Ч
	Ort::Session	rpn_session;	// onnxruntime_rpn
}AVS_ONNX_FILTER;

#endif