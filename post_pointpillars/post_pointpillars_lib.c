#include "post_pointpillars.h"
#include "post_pointpillars_lib.h"

/**********		box回归		**********/
// params:
//		anchor			I	预设anchor
//		preds			I	模型回归值
//		box_preds		O	真实值
// 备注：
void decode_box(float *anchor, float *preds, float *box_preds) {
	int i = 0;
	for (i = 0; i < AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1; i += 7) {
		float xa = anchor[i], ya = anchor[i + 1], za = anchor[i + 2], wa = anchor[i + 3],
			la = anchor[i + 4], ha = anchor[i + 5], ra = anchor[i + 6];
		float xt = preds[i], yt = preds[i + 1], zt = preds[i + 2], wt = preds[i + 3],
			lt = preds[i + 4], ht = preds[i + 5], rt = preds[i + 6];
		float za_top = za + ha / 2;
		float diag = sqrt(la * la + wa * wa);
		float xg = xt * diag + xa;
		float yg = yt * diag + ya;
		float zg_mid = zt * ha + za_top;
		float lg = exp(lt) * la;
		float wg = exp(wt) * wa;
		float hg = exp(ht) * ha;
		float rg = rt + ra;
		float zg = zg_mid - hg / 2;
		box_preds[i]		= xg;
		box_preds[i + 1]	= yg;
		box_preds[i + 2]	= zg;
		box_preds[i + 3]	= wg;
		box_preds[i + 4]	= lg;
		box_preds[i + 5]	= hg;
		box_preds[i + 6]	= rg;
	}
	printf("decode box done!\n");
}

// 取得最大值的下标和sigmoid值
void get_cls_info(float *cls_preds, int cls_num, int index, int *cls, float *prob) {
	int j = 0;
	float score = 0.f;
	for (j = 0; j < cls_num; ++j) {
		score = 1 / (1 + exp(-1.f * cls_preds[cls_num * index + j]));	// sigmoid
		if (score > *prob) {
			*prob	= score;
			*cls	= j;
		}
	}
}

// 取得最大值的下标
void get_dir_info(float *cls_preds, int cls_num, int index, int *dir) {
	int j = 0;
	float score = cls_preds[cls_num * index];
	for (j = 0; j < cls_num; ++j) {
		if (cls_preds[cls_num * index + j] > score) {
			score = cls_preds[cls_num * index + j];
			*dir = j;
		}
	}
}

void filter_process(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int		i				= 0;
	float	*anchor_mask	= inbuf->post_in.anchor_mask;
	float	*box_preds		= filter->box_preds;
	float	*cls_preds		= inbuf->rpn_out.out_185;
	float	*dir_preds		= inbuf->rpn_out.out_187;

	float	*mask_box_preds		= filter->mask_box_preds;
	int		*mask_cls			= filter->mask_cls;
	float	*mask_sigmoid		= filter->mask_sigmoid;
	int		*mask_dir_labels	= filter->mask_dir_labels;
	filter->mask_num = 0;
	for (i = 0; i < AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM; ++i) {
		int is_selected = anchor_mask[i];
		AVS_CHECK_CONTINUE(!is_selected);	// mask 过滤

		// 计算cls的sigmoid	1 / (1 + exp(-x)); 确定类别和置信度
		int		cls		= 0;
		float	prob	= 0.f;
		get_cls_info(cls_preds, AVS_LADAR_OBJ_END, i, &cls, &prob);
		AVS_CHECK_CONTINUE(prob < CONF_THRESHOLD);	// 置信度过滤
		AVS_CHECK_BREAK(filter->mask_num >= PRE_MAX_SIZE);	// nms上限

		// 得到dir信息
		int dir = 0;
		get_dir_info(dir_preds, AVS_LADAR_OBJ_END, i, &dir);

		// 根据mask保存信息
		memcpy(&mask_box_preds[AVS_ANCHOR_W * filter->mask_num], &box_preds[AVS_ANCHOR_W * i], AVS_ANCHOR_W * sizeof(float));
		mask_cls[filter->mask_num]			= cls;
		mask_sigmoid[filter->mask_num]		= prob;
		mask_dir_labels[filter->mask_num]	= dir;
		++filter->mask_num;
	}
	printf("filter_process done!\n");
}

void get_box_2d(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0;
	float *mask_box_preds	= filter->mask_box_preds;
	float *box_2d			= filter->box_2d;
	for (i = 0; i < filter->mask_num; ++i) {
		float x = mask_box_preds[AVS_ANCHOR_W * i + 0];
		float y = mask_box_preds[AVS_ANCHOR_W * i + 1];
		float h = mask_box_preds[AVS_ANCHOR_W * i + 3];
		float w = mask_box_preds[AVS_ANCHOR_W * i + 4];
		// center_to_corner_box2d
		float corners[4][2] = {
			{-1.f * CENTER_RATIO * h, -1.f * CENTER_RATIO * w},
			{-1.f * CENTER_RATIO * h, +1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * h, +1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * h, -1.f * CENTER_RATIO * w},
		};
		float angles = mask_box_preds[AVS_ANCHOR_W * i + 6];
		float rot_sin = sin(angles), rot_cos = cos(angles);
		float rot_mat[2][2] = { 
			{rot_cos, -rot_sin},
			{rot_sin, rot_cos},
		};
		float rot_corners_center[4][2] = { // 4x2 * 2x2
			{	corners[0][0] * rot_mat[0][0] + corners[0][1] * rot_mat[1][0] + x, 
				corners[0][0] * rot_mat[0][1] + corners[0][1] * rot_mat[1][1] + y},
			{	corners[1][0] * rot_mat[0][0] + corners[1][1] * rot_mat[1][0] + x,
				corners[1][0] * rot_mat[0][1] + corners[1][1] * rot_mat[1][1] + y},
			{	corners[2][0] * rot_mat[0][0] + corners[2][1] * rot_mat[1][0] + x,
				corners[2][0] * rot_mat[0][1] + corners[2][1] * rot_mat[1][1] + y},
			{	corners[3][0] * rot_mat[0][0] + corners[3][1] * rot_mat[1][0] + x,
				corners[3][0] * rot_mat[0][1] + corners[3][1] * rot_mat[1][1] + y},
		};
		// corner_to_standup_nd
		float x_min = AVS_MIN(rot_corners_center[0][0], AVS_MIN(rot_corners_center[1][0], 
			AVS_MIN(rot_corners_center[2][0], rot_corners_center[3][0])));
		float y_min = AVS_MIN(rot_corners_center[0][1], AVS_MIN(rot_corners_center[1][1],
			AVS_MIN(rot_corners_center[2][1], rot_corners_center[3][1])));
		float x_max = AVS_MAX(rot_corners_center[0][0], AVS_MAX(rot_corners_center[1][0],
			AVS_MAX(rot_corners_center[2][0], rot_corners_center[3][0])));
		float y_max = AVS_MAX(rot_corners_center[0][1], AVS_MAX(rot_corners_center[1][1],
			AVS_MAX(rot_corners_center[2][1], rot_corners_center[3][1])));
		box_2d[AVS_ANCHOR_NUM * i + 0] = x_min;
		box_2d[AVS_ANCHOR_NUM * i + 1] = y_min;
		box_2d[AVS_ANCHOR_NUM * i + 2] = x_max;
		box_2d[AVS_ANCHOR_NUM * i + 3] = y_max;
	}
}

// 位置交换 int 
void anchor_swap_int(int *a, int *b, int len) {
	memset(debug_cache, 0, 1024 * sizeof(int));
	memcpy(debug_cache, a, len);
	memcpy(a, b, len);
	memcpy(b, debug_cache, len);
}

// 位置交换 float
void anchor_swap_flt(float *a, float *b, int len) {
	memset(debug_cache, 0, 1024 * sizeof(float));
	memcpy(debug_cache, a, len);
	memcpy(a, b, len);
	memcpy(b, debug_cache, len);
}

void box_sort(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	int		*cls		= filter->mask_cls;
	int		*dir_lab	= filter->mask_dir_labels;
	float	*score		= filter->mask_sigmoid;
	float	*box_pred	= filter->mask_box_preds;
	float	*box_2d		= filter->box_2d;
	int		*nms_valid	= filter->nms_valid;
	for (i = 0; i < filter->mask_num; ++i) {
		nms_valid[i] = 1;
		for (j = i + 1; j < filter->mask_num; ++j) {
			if (score[i] < score[j]) {
				anchor_swap_int(&cls[i], &cls[j], sizeof(int));
				anchor_swap_int(&dir_lab[i], &dir_lab[j], sizeof(int));
				anchor_swap_flt(&score[i], &score[j], sizeof(float));
				anchor_swap_flt(&box_pred[AVS_ANCHOR_W * i], &box_pred[AVS_ANCHOR_W * j], AVS_ANCHOR_W * sizeof(float));
				anchor_swap_flt(&box_2d[AVS_ANCHOR_NUM * i], &box_2d[AVS_ANCHOR_NUM * j], AVS_ANCHOR_NUM * sizeof(float));
			}
		}
	}
}

// 矩形框iou计算
float anchor_iou(float *dst, float *src) {
	float x1 = AVS_MAX(dst[0], src[0]);
	float y1 = AVS_MAX(dst[1], src[1]);
	float x2 = AVS_MIN(dst[2], src[2]);
	float y2 = AVS_MIN(dst[3], src[3]);
	float w = AVS_MAX(x2 - x1 + 1, 0.f), h = AVS_MAX(y2 - y1 + 1, 0.f);
	float cross = w * h;
	float area1 = (src[2] - src[0] + 1) * (src[3] - src[1] + 1);
	float area2 = (dst[2] - dst[0] + 1) * (dst[3] - dst[1] + 1);
	float iou = cross / (area1 + area2 - cross + FLT_MIN);
	return iou;
}

void nms_process(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	float iou = 0.f;
	float	*score		= filter->mask_sigmoid;
	float	*box_2d		= filter->box_2d;
	int		*nms_valid	= filter->nms_valid;
	for (i = 0; i < filter->mask_num; ++i) {
		AVS_CHECK_CONTINUE(nms_valid[i] != 1);	// 无效anchor不对其他anchor进行nms
		for (j = i + 1; j < filter->mask_num; ++j) {
			AVS_CHECK_CONTINUE(nms_valid[j] != 1);	// 无效anchor不对其他anchor进行nms
			iou = anchor_iou(&box_2d[AVS_ANCHOR_NUM * i], &box_2d[AVS_ANCHOR_NUM * j]);
			if (iou > NMS_THRESHOLD) {
				nms_valid[j] = 0;
			}
		}
	}
}

void nms_post_out(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	int		*cls = filter->mask_cls;
	int		*dir_lab = filter->mask_dir_labels;
	float	*score = filter->mask_sigmoid;
	float	*box_pred = filter->mask_box_preds;
	float	*box_2d = filter->box_2d;
	int		*nms_valid = filter->nms_valid;
	for (j = 0; j < filter->mask_num; ++j) {
		if (nms_valid[j]) {
			//printf("%d\n", j);
			memcpy(&cls[i], &cls[j], sizeof(int));
			memcpy(&dir_lab[i], &dir_lab[j], sizeof(int));
			memcpy(&score[i], &score[j], sizeof(float));
			memcpy(&box_pred[AVS_ANCHOR_W * i], &box_pred[AVS_ANCHOR_W * j], AVS_ANCHOR_W * sizeof(float));
			memcpy(&box_2d[AVS_ANCHOR_NUM * i], &box_2d[AVS_ANCHOR_NUM * j], sizeof(float));
			++i;
		}
	}
	filter->mask_num = AVS_MIN(i, POST_MAX_SIZE);
}

void direction_classifier(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	int		*dir_lab = filter->mask_dir_labels;
	float	*box_pred = filter->mask_box_preds;
	for (i = 0; i < filter->mask_num; ++i) {
		box_pred[AVS_ANCHOR_W * i + 6] += ((box_pred[AVS_ANCHOR_W * i + 6] > 0) ^ (dir_lab[i] > 0)) * AVS_PI;
		//printf("%f\n", box_pred[AVS_ANCHOR_W * i + 6]);
	}
}

// m1[m][k] m2[k][n] m3[m][n]
void matrix_dot(float **m1, float **m2, int m, int k, int n, float **m3) {
	int i = 0, j = 0, l = 0;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			*((float *)m3 + i * n + j) = 0.f;
			for(l = 0; l < k; ++l){
				*((float *)m3 + i * n + j) += *((float *)m1 + i * k + l) * *((float *)m2 + l * n + j);
			}
		}
	}
}

// m1[m][n] m2[n][m]
void matrix_transpose(float **m1, int m, int n, float **m2) {
	int i = 0, j = 0;
	for (i = 0; i < m; ++i) {
		for (j = 0; j < n; ++j) {
			*((float *)m2 + j * m + i) = *((float *)m1 + i * n + j);
		}
	}
}

// 相机参数的仿射变换
void box_lidar_to_camera(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	float rect_trv2c[4][4]		= { 0.f };
	float rect_trv2c_t[4][4]	= { 0.f };
	float *box_pred				= filter->box_preds_camera;
	memcpy(filter->box_preds_camera, filter->mask_box_preds, PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float));
	matrix_dot(inbuf->post_in.post_rect, inbuf->post_in.trv2c, 4, 4, 4, rect_trv2c);
	matrix_transpose(rect_trv2c, 4, 4, rect_trv2c_t);
	for (i = 0; i < filter->mask_num; ++i) {
		float points[1][4] = { 1.f };
		float points_c[1][4] = { 0.f };
		points[0][0] = box_pred[AVS_ANCHOR_W * i + 0];
		points[0][1] = box_pred[AVS_ANCHOR_W * i + 1];
		points[0][2] = box_pred[AVS_ANCHOR_W * i + 2];
		//printf("%f %f %f \n", points[0][0], points[0][1], points[0][2]);
		matrix_dot(points, rect_trv2c_t, 1, 4, 4, points_c);
		box_pred[AVS_ANCHOR_W * i + 0] = points_c[0][0];
		box_pred[AVS_ANCHOR_W * i + 1] = points_c[0][1];
		box_pred[AVS_ANCHOR_W * i + 2] = points_c[0][2];
		//printf("%f %f %f \n", points_c[0][0], points_c[0][1], points_c[0][2]);
	}
}

void get_box_3d(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	float *box_pred = filter->box_preds_camera;
	float *box_2d = filter->box_preds_2d;
	float p2_transpose[4][4] = { 0.f };
	matrix_transpose(inbuf->post_in.p2, 4, 4, p2_transpose);
	for (i = 0; i < filter->mask_num; ++i) {	// l h w
		float x = box_pred[AVS_ANCHOR_W * i + 0];
		float y = box_pred[AVS_ANCHOR_W * i + 1];
		float z = box_pred[AVS_ANCHOR_W * i + 2];
		float l = box_pred[AVS_ANCHOR_W * i + 4];
		float h = box_pred[AVS_ANCHOR_W * i + 5];
		float w = box_pred[AVS_ANCHOR_W * i + 3];
		// center_to_corner_box3d
		float corners[8][3] = {
			{-1.f * CENTER_RATIO * l, -1.f * CENTER_RATIO_H * h, -1.f * CENTER_RATIO * w},
			{-1.f * CENTER_RATIO * l, -1.f * CENTER_RATIO_H * h, +1.f * CENTER_RATIO * w},
			{-1.f * CENTER_RATIO * l, +0.f * CENTER_RATIO_H * h, +1.f * CENTER_RATIO * w},
			{-1.f * CENTER_RATIO * l, +0.f * CENTER_RATIO_H * h, -1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * l, -1.f * CENTER_RATIO_H * h, -1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * l, -1.f * CENTER_RATIO_H * h, +1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * l, +0.f * CENTER_RATIO_H * h, +1.f * CENTER_RATIO * w},
			{+1.f * CENTER_RATIO * l, +0.f * CENTER_RATIO_H * h, -1.f * CENTER_RATIO * w},
		};
		float angles = box_pred[AVS_ANCHOR_W * i + 6];
		float rot_sin = sin(angles), rot_cos = cos(angles);
		float rot_mat[3][3] = {
			{rot_cos, 0.f, -rot_sin},
			{0.f, 1.f, 0.f},
			{rot_sin, 0.f, rot_cos},
		};
		// 8x3 * 3x3
		float rot_corners_center[8][4] = {
			{	corners[0][0] * rot_mat[0][0] + corners[0][1] * rot_mat[1][0] + corners[0][2] * rot_mat[2][0] + x,
				corners[0][0] * rot_mat[0][1] + corners[0][1] * rot_mat[1][1] + corners[0][2] * rot_mat[2][1] + y,
				corners[0][0] * rot_mat[0][2] + corners[0][1] * rot_mat[1][2] + corners[0][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[1][0] * rot_mat[0][0] + corners[1][1] * rot_mat[1][0] + corners[1][2] * rot_mat[2][0] + x,
				corners[1][0] * rot_mat[0][1] + corners[1][1] * rot_mat[1][1] + corners[1][2] * rot_mat[2][1] + y,
				corners[1][0] * rot_mat[0][2] + corners[1][1] * rot_mat[1][2] + corners[1][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[2][0] * rot_mat[0][0] + corners[2][1] * rot_mat[1][0] + corners[2][2] * rot_mat[2][0] + x,
				corners[2][0] * rot_mat[0][1] + corners[2][1] * rot_mat[1][1] + corners[2][2] * rot_mat[2][1] + y,
				corners[2][0] * rot_mat[0][2] + corners[2][1] * rot_mat[1][2] + corners[2][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[3][0] * rot_mat[0][0] + corners[3][1] * rot_mat[1][0] + corners[3][2] * rot_mat[2][0] + x,
				corners[3][0] * rot_mat[0][1] + corners[3][1] * rot_mat[1][1] + corners[3][2] * rot_mat[2][1] + y,
				corners[3][0] * rot_mat[0][2] + corners[3][1] * rot_mat[1][2] + corners[3][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[4][0] * rot_mat[0][0] + corners[4][1] * rot_mat[1][0] + corners[4][2] * rot_mat[2][0] + x,
				corners[4][0] * rot_mat[0][1] + corners[4][1] * rot_mat[1][1] + corners[4][2] * rot_mat[2][1] + y,
				corners[4][0] * rot_mat[0][2] + corners[4][1] * rot_mat[1][2] + corners[4][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[5][0] * rot_mat[0][0] + corners[5][1] * rot_mat[1][0] + corners[5][2] * rot_mat[2][0] + x,
				corners[5][0] * rot_mat[0][1] + corners[5][1] * rot_mat[1][1] + corners[5][2] * rot_mat[2][1] + y,
				corners[5][0] * rot_mat[0][2] + corners[5][1] * rot_mat[1][2] + corners[5][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[6][0] * rot_mat[0][0] + corners[6][1] * rot_mat[1][0] + corners[6][2] * rot_mat[2][0] + x,
				corners[6][0] * rot_mat[0][1] + corners[6][1] * rot_mat[1][1] + corners[6][2] * rot_mat[2][1] + y,
				corners[6][0] * rot_mat[0][2] + corners[6][1] * rot_mat[1][2] + corners[6][2] * rot_mat[2][2] + z,
				0.f},
			{	corners[7][0] * rot_mat[0][0] + corners[7][1] * rot_mat[1][0] + corners[7][2] * rot_mat[2][0] + x,
				corners[7][0] * rot_mat[0][1] + corners[7][1] * rot_mat[1][1] + corners[7][2] * rot_mat[2][1] + y,
				corners[7][0] * rot_mat[0][2] + corners[7][1] * rot_mat[1][2] + corners[7][2] * rot_mat[2][2] + z,
				0.f},
		};
		float point_2d[8][4] = {0.f};
		// project_to_image
		matrix_dot(rot_corners_center, p2_transpose, 8, 4, 4, point_2d);
		for (j = 0; j < 8; ++j) {
			point_2d[j][0] /= point_2d[j][2];
			point_2d[j][1] /= point_2d[j][2];
		}
		box_2d[AVS_ANCHOR_W * i + 0] = AVS_MIN(point_2d[0][0], AVS_MIN(point_2d[1][0], 
			AVS_MIN(point_2d[2][0], AVS_MIN(point_2d[3][0], AVS_MIN(point_2d[4][0], 
				AVS_MIN(point_2d[5][0], AVS_MIN(point_2d[6][0], point_2d[7][0])))))));
		box_2d[AVS_ANCHOR_W * i + 1] = AVS_MIN(point_2d[0][1], AVS_MIN(point_2d[1][1],
			AVS_MIN(point_2d[2][1], AVS_MIN(point_2d[3][1], AVS_MIN(point_2d[4][1],
				AVS_MIN(point_2d[5][1], AVS_MIN(point_2d[6][1], point_2d[7][1])))))));
		box_2d[AVS_ANCHOR_W * i + 2] = AVS_MAX(point_2d[0][0], AVS_MAX(point_2d[1][0],
			AVS_MAX(point_2d[2][0], AVS_MAX(point_2d[3][0], AVS_MAX(point_2d[4][0],
				AVS_MAX(point_2d[5][0], AVS_MAX(point_2d[6][0], point_2d[7][0])))))));
		box_2d[AVS_ANCHOR_W * i + 3] = AVS_MAX(point_2d[0][1], AVS_MAX(point_2d[1][1],
			AVS_MAX(point_2d[2][1], AVS_MAX(point_2d[3][1], AVS_MAX(point_2d[4][1],
				AVS_MAX(point_2d[5][1], AVS_MAX(point_2d[6][1], point_2d[7][1])))))));
	}
}

void get_output_info(POINTPILLARS_POST_INFO *inbuf, AVS_POST_FILTER *filter) {
	int i = 0, j = 0;
	float	*mask_box_preds		= filter->mask_box_preds;
	float	*box_preds_camera	= filter->box_preds_camera;
	float	*box_preds_2d		= filter->box_preds_2d;
	float	*conf				= filter->mask_sigmoid;
	int		*cls				= filter->mask_cls;
	inbuf->post_out.obj_num = filter->mask_num;
	for (i = 0; i < inbuf->post_out.obj_num; ++i) {
		inbuf->post_out.lidar_obj[i].box_preds.x = mask_box_preds[i * AVS_ANCHOR_W + 0];
		inbuf->post_out.lidar_obj[i].box_preds.y = mask_box_preds[i * AVS_ANCHOR_W + 1];
		inbuf->post_out.lidar_obj[i].box_preds.z = mask_box_preds[i * AVS_ANCHOR_W + 2];
		inbuf->post_out.lidar_obj[i].box_preds.w = mask_box_preds[i * AVS_ANCHOR_W + 3];
		inbuf->post_out.lidar_obj[i].box_preds.l = mask_box_preds[i * AVS_ANCHOR_W + 4];
		inbuf->post_out.lidar_obj[i].box_preds.h = mask_box_preds[i * AVS_ANCHOR_W + 5];
		inbuf->post_out.lidar_obj[i].box_preds.r = mask_box_preds[i * AVS_ANCHOR_W + 6];

		inbuf->post_out.lidar_obj[i].box_preds_camera.x = box_preds_camera[i * AVS_ANCHOR_W + 0];
		inbuf->post_out.lidar_obj[i].box_preds_camera.y = box_preds_camera[i * AVS_ANCHOR_W + 1];
		inbuf->post_out.lidar_obj[i].box_preds_camera.z = box_preds_camera[i * AVS_ANCHOR_W + 2];
		inbuf->post_out.lidar_obj[i].box_preds_camera.w = box_preds_camera[i * AVS_ANCHOR_W + 3];
		inbuf->post_out.lidar_obj[i].box_preds_camera.l = box_preds_camera[i * AVS_ANCHOR_W + 4];
		inbuf->post_out.lidar_obj[i].box_preds_camera.h = box_preds_camera[i * AVS_ANCHOR_W + 5];
		inbuf->post_out.lidar_obj[i].box_preds_camera.r = box_preds_camera[i * AVS_ANCHOR_W + 6];

		inbuf->post_out.lidar_obj[i].box_preds_2d.x1 = box_preds_2d[i * AVS_ANCHOR_NUM + 0];
		inbuf->post_out.lidar_obj[i].box_preds_2d.y1 = box_preds_2d[i * AVS_ANCHOR_NUM + 1];
		inbuf->post_out.lidar_obj[i].box_preds_2d.x2 = box_preds_2d[i * AVS_ANCHOR_NUM + 2];
		inbuf->post_out.lidar_obj[i].box_preds_2d.y2 = box_preds_2d[i * AVS_ANCHOR_NUM + 3];

		inbuf->post_out.lidar_obj[i].conf = conf[i];

		inbuf->post_out.lidar_obj[i].cls = cls[i];

		AVS_CHECK_BREAK(i >= POST_MAX_SIZE);
	}
}