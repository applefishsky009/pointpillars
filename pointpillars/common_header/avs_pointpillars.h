#ifndef _AVS_POINTPILLARS_H_
#define _AVS_POINTPILLARS_H_

#ifdef __cplusplus
extern "C" {
#endif

/********** ONNX所用结构体 **********/
#define AVS_PREONNX_MAX_IN_C		12000
#define AVS_PREONNX_MAX_DIM			100
#define AVS_PREONNX_OUT_C			64

#define AVS_RPNONNX_IN_C			64
#define AVS_RPNONNX_IN_H			496
#define AVS_RPNONNX_IN_W			432
#define AVS_RPNONNX_OUT_1			28
#define AVS_RPNONNX_OUT_2			8

#define AVS_ANCHOR_NUM				4
#define AVS_ANCHOR_W				7

#define AVS_CLS_NUM					2

typedef struct _AVS_PP_PREPROCESS_OUT_ {
	float *pillar_x;	// 1x1x12000x100
	float *pillar_y;	// 1x1x12000x100
	float *pillar_z;	// 1x1x12000x100
	float *pillar_i;	// 1x1x12000x100
	float *num_voxels;	// 1x12000
	float *x_sub_shaped;	// 1x1x12000x100
	float *y_sub_shaped;	// 1x1x12000x100
	float *mask;
}AVS_PP_PREPROCESS_OUT;

typedef struct _AVS_PP_SCATTER_IN_ {
	float *pillar_feature;	// 1x64x12000x1
}AVS_PP_SCATTER_IN;

typedef struct _AVS_PP_RPN_IN_ {
	float *spatial_features;	// 1x64x496x432
}AVS_PP_RPN_IN;

typedef struct _AVS_PP_RPN_OUT_ {
	float *out_184;	//  1x496x462x28	box
	float *out_185; //	1x496x462x8		cls
	float *out_187; //	1x496x462x8		direction
}AVS_PP_RPN_OUT;

/********** POST所用结构体 **********/
typedef struct _AVS_POST_IN_ {
	float			post_rect[4][4];
	float			trv2c[4][4];
	float			p2[4][4];
	float			*anchors;		// w * h * 4	max(496 x 432 x 4)
	float			*anchor_mask;	// w * h		max(496 x 432)
}AVS_POST_IN;

#ifdef __cplusplus
}
#endif

#endif