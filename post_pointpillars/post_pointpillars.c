#include "post_pointpillars.h"
#include "post_pointpillars_lib.h"
#include "post_pointpillars_common_lib.h"

unsigned int PointPillars_Post_GetMemSize(		POINTPILLARS_POST_INFO		*input,
												AVS_MEM_TAB					mem_tab[POINTPILALRS_POST_MEM_TAB]){
	unsigned int	mem_size	= 0;
	AVS_POST_IN		*post_input	= &input->post_in;

	// anchors内存，位于input			w * h * 4	max(496 x 432 x 4)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// anchor_mask内存，位于input		w * h		max(496 x 432)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// rpn.onnx输出内存 位于input
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// post 算法库句柄内存
	mem_size += sizeof(AVS_POST_FILTER) + AVS_MEM_ALIGN_128BYTE;
	// 预测的box 位于filter	496x432x4x7(x,y,z,w,l,h,r)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+置信度过滤之后的box, 位于filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+置信度度过滤之后的类别, 位于filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// mask+置信度度过滤之后的置信度, 位于filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+置信度度过滤之后的dir, 位于filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// nms前处理得到的box_2d
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// nms所用的valid_flag, 位于filter
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// 最终输出的激光雷达坐标转化为相机坐标系所用内存, 位于filter
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// 相机坐标系的2D输出所用内存, 位于filter
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	mem_tab[0].size = mem_size;
	mem_tab[0].base = NULL;
	mem_tab[0].alignment = AVS_MEM_ALIGN_128BYTE;
	printf("GetMemsize Done!\n");
	return 0;
}

unsigned int PointPillars_Post_CreatMemSize(	POINTPILLARS_POST_INFO		*input,
												AVS_MEM_TAB					mem_tab[POINTPILALRS_POST_MEM_TAB],
												void						**handle) {
	int i						= 0;
	AVS_POST_FILTER				*filter		= NULL;
	AVS_POST_IN					*post_in	= &input->post_in;
	AVS_PP_RPN_OUT				*rpn_out	= &input->rpn_out;
	AVS_POST_POINTPILLARS_BUF	mem_buf;
	mem_buf.start				= mem_tab[0].base;
	mem_buf.cur_pos				= mem_tab[0].base;
	mem_buf.end					= (void *)((QWORD)mem_buf.cur_pos + (QWORD)mem_tab[0].size);

	// anchors内存，位于input			w * h * 4	max(496 x 432 x 4)
	post_in->anchors = (unsigned char *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * AVS_ANCHOR_W * sizeof(float));
	// anchor_mask内存，位于input		w * h		max(496 x 432)
	post_in->anchor_mask = (unsigned char *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float));

	// rpn.onnx输出内存
	rpn_out->out_184 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));	// 	184
	AVS_CHECK_ERROR(rpn_out->out_184 == NULL, AVS_LIB_PTR_NULL);
	rpn_out->out_185 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	185
	AVS_CHECK_ERROR(rpn_out->out_185 == NULL, AVS_LIB_PTR_NULL);
	rpn_out->out_187 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	187
	AVS_CHECK_ERROR(rpn_out->out_187 == NULL, AVS_LIB_PTR_NULL);

	// 算法库句柄内存
	filter = (AVS_POST_FILTER *)AVS_POST_COM_alloc_buffer(&mem_buf, sizeof(AVS_POST_FILTER));
	AVS_CHECK_ERROR(filter == NULL, AVS_LIB_PTR_NULL);
	// 预测的box 位于filter	496x432x4x7(x,y,z,w,l,h,r)
	filter->box_preds = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));	// 	184
	AVS_CHECK_ERROR(filter->box_preds == NULL, AVS_LIB_PTR_NULL);
	// mask+置信度过滤之后的box, 位于filter
	filter->mask_box_preds = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));
	AVS_CHECK_ERROR(filter->mask_box_preds == NULL, AVS_LIB_PTR_NULL);
	// mask+置信度度过滤之后的类别, 位于filter
	filter->mask_cls = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int));
	AVS_CHECK_ERROR(filter->mask_cls == NULL, AVS_LIB_PTR_NULL);
	// mask+置信度度过滤之后的置信度, 位于filter
	filter->mask_sigmoid = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(float));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float));
	AVS_CHECK_ERROR(filter->mask_sigmoid == NULL, AVS_LIB_PTR_NULL);
	// mask+置信度度过滤之后的dir, 位于filter
	filter->mask_dir_labels = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int));
	AVS_CHECK_ERROR(filter->mask_dir_labels == NULL, AVS_LIB_PTR_NULL);
	// nms前处理得到的box_2d
	filter->box_2d = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float));
	AVS_CHECK_ERROR(filter->box_2d == NULL, AVS_LIB_PTR_NULL);
	// nms所用的valid_flag, 位于filter
	filter->nms_valid = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
	AVS_CHECK_ERROR(filter->nms_valid == NULL, AVS_LIB_PTR_NULL);
	// 最终输出的激光雷达坐标转化为相机坐标系所用内存, 位于filter
	filter->box_preds_camera = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float));
	AVS_CHECK_ERROR(filter->box_preds_camera == NULL, AVS_LIB_PTR_NULL);
	// 相机坐标系的2D输出所用内存, 位于filter
	filter->box_preds_2d = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float));
	AVS_CHECK_ERROR(filter->box_preds_2d == NULL, AVS_LIB_PTR_NULL);

	*handle = (void *)filter;
	printf("CreatMemSize Done!\n");
	return 0;
}

unsigned int PointPillars_Post_Init(			void						*handle,
												POINTPILLARS_POST_INFO		*inbuf,
												int							in_buf_size) {
	printf("Init Done!\n");
	return 0;
}

unsigned int PointPillars_Post_Proc(			void						*handle,
												POINTPILLARS_POST_INFO		*inbuf,
												int							in_buf_size,
												POINTPILLARS_POST_INFO		*outbuf,
												int							out_buf_size) {
	
	AVS_POST_FILTER	*filter = (AVS_POST_FILTER *)handle;
	
	// decode box pred	anchor + out_184	xyzwlhr
	decode_box(inbuf->post_in.anchors, inbuf->rpn_out.out_184, filter->box_preds); // anchor回归

	// mask 过滤  类别sigmoid计算  方向indices计算 置信度过滤
	// 该步骤可以和decode合并以节省内存，这里使用和python相同的计算过程, 未合并
	filter_process(inbuf, filter);	// nms前处理 - 过滤

	// box_2d 0 1 3 4 6 = x y l h r
	get_box_2d(inbuf, filter);	// nms前处理 - box_2d
	
	// 根据置信度排序
	box_sort(inbuf, filter);

	// nms
	nms_process(inbuf, filter);	// nms

	// nms out 
	nms_post_out(inbuf, filter);	// nms后处理 - final_scores final_labels

	// nms out 
	direction_classifier(inbuf, filter); // 更新r final_box_preds

	// box_lidar_to_camera
	box_lidar_to_camera(inbuf, filter);	// xyzwlhr

	// box_3d
	get_box_3d(inbuf, filter);

	printf("Proc Done!\n");
	return 0;
}