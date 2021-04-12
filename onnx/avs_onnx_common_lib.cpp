#include "avs_onnx_common_lib.h"

void *AVS_ONNX_COM_alloc_buffer(AVS_ONNX_BUF *avs_buf, int size) {
	void *buf;
	int free_size;
	// �����п����ڴ����ʼλ��
	buf = (void *)(((QWORD)avs_buf->cur_pos + (AVS_MEM_ALIGN_128BYTE - 1)) & (~(AVS_MEM_ALIGN_128BYTE - 1)));
	// ���㻺����ʣ��ռ��С
	free_size = (QWORD)avs_buf->end - (QWORD)buf;
	// �ռ䲻�������ؿ�ָ��
	if (free_size < size) {
		buf = NULL;
	}
	else {
		// ��շ����ڴ�
		memset(buf, 0, size);
		// ���¿���ָ��λ��
		avs_buf->cur_pos = (void *)((QWORD)buf + size);
	}
	return buf;
}
