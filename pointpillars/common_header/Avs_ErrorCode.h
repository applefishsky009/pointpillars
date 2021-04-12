#ifndef _AVS_ERRORCODE_H_
#define _AVS_ERRORCODE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define AVS_LIB_OK						0x00000000	// �ɹ�
#define AVS_LIB_MEM_OUT					0x60000000	// �ڴ治��
#define AVS_LIB_RESOLUTION_UNSUPPORT	0x60000001	// �ֱ��ʲ�֧��
#define AVS_LIB_PTR_NULL				0x60000002	// ����ָ��Ϊ��
#define AVS_LIB_KEY_PARAM_ERR			0x60000003	// �߼��������ô���

#define AVS_CHECK_ERROR(state, error_code)	\
if (state) {								\
	printf("[AVS Printf] %s-%d-%s %x\n", __FILE__, __LINE__, __FUNCTION__, error_code);	\
	return error_code;							\
}

#define AVS_CHECK_CONTINUE(state)	\
if (state) {						\
	continue;						\
}

#define AVS_CHECK_RETURN(state, error_code)	\
if (state) {								\
	return error_code;						\
}

#define AVS_CHECK_BREAK(state)	\
if (state) {					\
	break;						\
}

#ifdef __cplusplus
}
#endif

#endif