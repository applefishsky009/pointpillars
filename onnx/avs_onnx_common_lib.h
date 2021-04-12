#ifndef _AVS_ONNX_COMMON_LIB_H_
#define _AVS_ONNX_COMMON_LIB_H_

#include "Avs_Base.h"
#include "Avs_Common.h"

typedef struct _AVS_ONNX_BUF_ {
	void *start;
	void *end;
	void *cur_pos;
}AVS_ONNX_BUF;

void *AVS_ONNX_COM_alloc_buffer(	AVS_ONNX_BUF *avs_buf,
									int size);

#endif
