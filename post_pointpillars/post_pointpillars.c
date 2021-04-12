#include "post_pointpillars.h"
#include "post_pointpillars_lib.h"
#include "post_pointpillars_common_lib.h"

unsigned int PointPillars_Post_GetMemSize(		POINTPILLARS_POST_INFO		*input,
												AVS_MEM_TAB					mem_tab[POINTPILALRS_POST_MEM_TAB]){
	unsigned int	mem_size	= 0;
	AVS_POST_IN		*post_input	= &input->post_in;

	// anchors�ڴ棬λ��input			w * h * 4	max(496 x 432 x 4)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// anchor_mask�ڴ棬λ��input		w * h		max(496 x 432)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// rpn.onnx����ڴ� λ��input
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// post �㷨�����ڴ�
	mem_size += sizeof(AVS_POST_FILTER) + AVS_MEM_ALIGN_128BYTE;
	// Ԥ���box λ��filter	496x432x4x7(x,y,z,w,l,h,r)
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+���Ŷȹ���֮���box, λ��filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+���Ŷȶȹ���֮������, λ��filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// mask+���Ŷȶȹ���֮������Ŷ�, λ��filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// mask+���Ŷȶȹ���֮���dir, λ��filter
	//mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// nmsǰ����õ���box_2d
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// nms���õ�valid_flag, λ��filter
	mem_size += PRE_MAX_SIZE * sizeof(int) + AVS_MEM_ALIGN_128BYTE;
	// ��������ļ����״�����ת��Ϊ�������ϵ�����ڴ�, λ��filter
	mem_size += PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	// �������ϵ��2D��������ڴ�, λ��filter
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

	// anchors�ڴ棬λ��input			w * h * 4	max(496 x 432 x 4)
	post_in->anchors = (unsigned char *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * AVS_ANCHOR_W * sizeof(float));
	// anchor_mask�ڴ棬λ��input		w * h		max(496 x 432)
	post_in->anchor_mask = (unsigned char *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float));

	// rpn.onnx����ڴ�
	rpn_out->out_184 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));	// 	184
	AVS_CHECK_ERROR(rpn_out->out_184 == NULL, AVS_LIB_PTR_NULL);
	rpn_out->out_185 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	185
	AVS_CHECK_ERROR(rpn_out->out_185 == NULL, AVS_LIB_PTR_NULL);
	rpn_out->out_187 = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	187
	AVS_CHECK_ERROR(rpn_out->out_187 == NULL, AVS_LIB_PTR_NULL);

	// �㷨�����ڴ�
	filter = (AVS_POST_FILTER *)AVS_POST_COM_alloc_buffer(&mem_buf, sizeof(AVS_POST_FILTER));
	AVS_CHECK_ERROR(filter == NULL, AVS_LIB_PTR_NULL);
	// Ԥ���box λ��filter	496x432x4x7(x,y,z,w,l,h,r)
	filter->box_preds = (float *)AVS_POST_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));	// 	184
	AVS_CHECK_ERROR(filter->box_preds == NULL, AVS_LIB_PTR_NULL);
	// mask+���Ŷȹ���֮���box, λ��filter
	filter->mask_box_preds = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));
	AVS_CHECK_ERROR(filter->mask_box_preds == NULL, AVS_LIB_PTR_NULL);
	// mask+���Ŷȶȹ���֮������, λ��filter
	filter->mask_cls = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int));
	AVS_CHECK_ERROR(filter->mask_cls == NULL, AVS_LIB_PTR_NULL);
	// mask+���Ŷȶȹ���֮������Ŷ�, λ��filter
	filter->mask_sigmoid = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(float));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(float));
	AVS_CHECK_ERROR(filter->mask_sigmoid == NULL, AVS_LIB_PTR_NULL);
	// mask+���Ŷȶȹ���֮���dir, λ��filter
	filter->mask_dir_labels = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
		//AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_ANCHOR_NUM * sizeof(int));
	AVS_CHECK_ERROR(filter->mask_dir_labels == NULL, AVS_LIB_PTR_NULL);
	// nmsǰ����õ���box_2d
	filter->box_2d = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_NUM * sizeof(float));
	AVS_CHECK_ERROR(filter->box_2d == NULL, AVS_LIB_PTR_NULL);
	// nms���õ�valid_flag, λ��filter
	filter->nms_valid = (int *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * sizeof(int));
	AVS_CHECK_ERROR(filter->nms_valid == NULL, AVS_LIB_PTR_NULL);
	// ��������ļ����״�����ת��Ϊ�������ϵ�����ڴ�, λ��filter
	filter->box_preds_camera = (float *)AVS_POST_COM_alloc_buffer(&mem_buf, PRE_MAX_SIZE * AVS_ANCHOR_W * sizeof(float));
	AVS_CHECK_ERROR(filter->box_preds_camera == NULL, AVS_LIB_PTR_NULL);
	// �������ϵ��2D��������ڴ�, λ��filter
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
	decode_box(inbuf->post_in.anchors, inbuf->rpn_out.out_184, filter->box_preds); // anchor�ع�

	// mask ����  ���sigmoid����  ����indices���� ���Ŷȹ���
	// �ò�����Ժ�decode�ϲ��Խ�ʡ�ڴ棬����ʹ�ú�python��ͬ�ļ������, δ�ϲ�
	filter_process(inbuf, filter);	// nmsǰ���� - ����

	// box_2d 0 1 3 4 6 = x y l h r
	get_box_2d(inbuf, filter);	// nmsǰ���� - box_2d
	
	// �������Ŷ�����
	box_sort(inbuf, filter);

	// nms
	nms_process(inbuf, filter);	// nms

	// nms out 
	nms_post_out(inbuf, filter);	// nms���� - final_scores final_labels

	// nms out 
	direction_classifier(inbuf, filter); // ����r final_box_preds

	// box_lidar_to_camera
	box_lidar_to_camera(inbuf, filter);	// xyzwlhr

	// box_3d
	get_box_3d(inbuf, filter);

	printf("Proc Done!\n");
	return 0;
}