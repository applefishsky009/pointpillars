#ifndef _AVS_DNN_H_
#define _AVS_DNN_H_

#include "Avs_Base.h"
#include "Avs_Common.h"
#include "Avs_ErrorCode.h"
#include "avs_onnx_lib.h"

void dnn_onnx_init(	dnn::Net		&net,
					char			*onnx);

void dnn_onnx_show(dnn::Net	&net);

void dnn_onnx_image_forward(	AVS_ONNX_FILTER *filter,
								char			*image);

void dnn_onnx_vector_forward(	dnn::Net		&net,
								AVS_PP_PREPROCESS_OUT	*inbu);

#endif