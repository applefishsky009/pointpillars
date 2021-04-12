#ifndef _AVS_ONNXRUNTIME_H_
#define _AVS_ONNXRUNTIME_H_

#include "Avs_Base.h"
#include "Avs_Common.h"
#include "Avs_ErrorCode.h"
#include "avs_onnx_lib.h"

// onnx pre模型初始化
void onnxruntime_init(	AVS_ONNX_FILTER *filter,
						char			*onnx);

// onnx 模型信息打印
void onnxruntime_show(	Ort::Session	&session);

// onnx pre模型前向
float* onnxruntime_foward(	Ort::Session			&session,
							AVS_PP_PREPROCESS_OUT	&inbuf);

// onnx pre模型输出
void onnxruntime_output(	float					*ort_inputs,
							AVS_PP_SCATTER_IN		&onnx_scatter_in);

// onnx rpn模型初始化
void onnxruntime_rpn_init(	AVS_ONNX_FILTER *filter,
							char			*rpn_onnx);

// onnx rpn前向
vector<float *> onnxruntime_rpn_foward(	Ort::Session			&session,
										AVS_PP_RPN_IN			&onnx_pre_input);

// onnx pre模型输出
void onnxruntime_rpn_output(	vector<float *>			&ort_outputs,
								AVS_PP_RPN_OUT			&onnx_rpn_out);

#endif