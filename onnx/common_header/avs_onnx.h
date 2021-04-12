#ifndef _AVS_ONNX_H_
#define _AVS_ONNX_H_

#include "Avs_Base.h"
#include "Avs_Common.h"
#include "Avs_ErrorCode.h"
#include "avs_pointpillars.h"

#define AVS_ONNX_MEM_TAB	1

typedef struct _AVS_ONNX_INFO_ {
	// onnx pre的数据
	char					*image;
	char					*onnx;
	AVS_PP_PREPROCESS_OUT	onnx_pre_input;
	AVS_PP_SCATTER_IN		onnx_scatter_in;
	// onnx rpn的数据
	char					*rpn_onnx;
	AVS_PP_RPN_IN			onnx_rpn_in;
	AVS_PP_RPN_OUT			onnx_rpn_out;
}AVS_ONNX_INFO;

unsigned int AVS_Onnx_GetMemSize(	AVS_ONNX_INFO		*input,
									AVS_MEM_TAB			mem_tab[AVS_ONNX_MEM_TAB]);

unsigned int AVS_Onnx_CreatMemSize(	AVS_ONNX_INFO		*input,
									AVS_MEM_TAB			mem_tab[AVS_ONNX_MEM_TAB],
									void				**handle);

unsigned int AVS_Onnx_Init(	void						*handle, 
							AVS_ONNX_INFO				*inbuf,
							int							in_buf_size);

unsigned int AVS_Onnx_Forward(	void						*handle,
								AVS_ONNX_INFO				*inbuf,
								int							in_buf_size,
								AVS_ONNX_INFO				*outbuf,
								int							out_buf_size);

unsigned int AVS_Onnx_Rpn_Forward(	void						*handle,
									AVS_ONNX_INFO				*inbuf,
									int							in_buf_size,
									AVS_ONNX_INFO				*outbuf,
									int							out_buf_size);

#endif
