#include "avs_onnx_common_lib.h"

void *AVS_ONNX_COM_alloc_buffer(AVS_ONNX_BUF *avs_buf, int size) {
	void *buf;
	int free_size;
	// 缓存中空余内存的起始位置
	buf = (void *)(((QWORD)avs_buf->cur_pos + (AVS_MEM_ALIGN_128BYTE - 1)) & (~(AVS_MEM_ALIGN_128BYTE - 1)));
	// 计算缓存中剩余空间大小
	free_size = (QWORD)avs_buf->end - (QWORD)buf;
	// 空间不够，返回空指针
	if (free_size < size) {
		buf = NULL;
	}
	else {
		// 清空分配内存
		memset(buf, 0, size);
		// 更新空余指针位置
		avs_buf->cur_pos = (void *)((QWORD)buf + size);
	}
	return buf;
}
