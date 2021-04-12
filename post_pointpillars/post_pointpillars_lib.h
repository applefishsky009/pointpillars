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
	float			*box_preds;			// 全部的anchor回归	496x432x28
	// mask 过滤 + 类别sigmoid计算  方向indices计算 置信度过滤
	unsigned int	mask_num;			
	float			*mask_box_preds;	// mask过滤后的坐标	最终输出的激光雷达坐标在这里
	int				*mask_cls;			// mask过滤后的类别	最终输出的类别在这里
	float			*mask_sigmoid;		// mask过滤后的置信度	最终输出的置信度在这里
	int				*mask_dir_labels;	// mask过滤后的dir	
	float			*box_2d;			// nms前处理数据
	int				*nms_valid;			// nms过滤所用flag
	float			*box_preds_camera;	// 最终输出的激光雷达坐标转化为相机坐标系
	float			*box_preds_2d;		// 相机坐标系的2D输出
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