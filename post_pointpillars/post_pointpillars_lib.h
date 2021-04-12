#ifndef _POST_POINTPILLARS_LIB_H_
#define _POST_POINTPILLARS_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>
#include <float.h>
#include "post_pointpillars_common_lib.h"

#define CONF_THRESHOLD	0.05
#define NMS_THRESHOLD	0.5
#define CENTER_RATIO	0.5
#define CENTER_RATIO_H	1
#define PRE_MAX_SIZE	1000
#define POST_MAX_SIZE	300
#define AVS_PI			3.141592653

float	debug_cache[1024];

typedef struct _AVS_POST_FILTER_ {
	float			*box_preds;			// ȫ����anchor�ع�	496x432x28
	// mask ���� + ���sigmoid����  ����indices���� ���Ŷȹ���
	unsigned int	mask_num;			
	float			*mask_box_preds;	// mask���˺������	��������ļ����״�����������
	int				*mask_cls;			// mask���˺�����	������������������
	float			*mask_sigmoid;		// mask���˺�����Ŷ�	������������Ŷ�������
	int				*mask_dir_labels;	// mask���˺��dir	
	float			*box_2d;			// nmsǰ��������
	int				*nms_valid;			// nms��������flag
	float			*box_preds_camera;	// ��������ļ����״�����ת��Ϊ�������ϵ
	float			*box_preds_2d;		// �������ϵ��2D���
}AVS_POST_FILTER;

void decode_box(float *anchor, float *preds, float *box_preds);

void filter_process(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void get_box_2d(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void box_sort(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void nms_process(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void nms_post_out(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void direction_classifier(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void box_lidar_to_camera(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

void get_box_3d(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter);

#ifdef __cplusplus
}
#endif

#endif