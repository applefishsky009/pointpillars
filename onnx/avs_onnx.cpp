#include "avs_onnx.h"
#include "avs_onnx_lib.h"
#include "avs_onnx_common_lib.h"
#include "avs_dnn.h"
#include "avs_onnxruntime.h"

unsigned int AVS_Onnx_GetMemSize(	AVS_ONNX_INFO		*input,
									AVS_MEM_TAB			mem_tab[AVS_ONNX_MEM_TAB])
{
	unsigned int mem_size = 0;

	// pre.onnx输入内存, 完整的lib应该在pointpillars接口中
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// pillar_x
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// pillar_y
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// pillar_z
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// pillar_i
	mem_size += AVS_PREONNX_MAX_IN_C * sizeof(float) + AVS_MEM_ALIGN_128BYTE;						// num_voxels
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// x_sub_shaped
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// y_sub_shaped
	mem_size += AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// mask

	// pre.onnx输出内存, 完整的lib应该在pointpillars接口中
	mem_size += AVS_PREONNX_OUT_C * AVS_PREONNX_MAX_IN_C * sizeof(float) + AVS_MEM_ALIGN_128BYTE;	// pillars feature

	// rpn.onnx输入内存
	mem_size += AVS_RPNONNX_IN_C * AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// rpn.onnx输出内存
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;
	mem_size += AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float) + AVS_MEM_ALIGN_128BYTE;

	// 算法库句柄内存
	mem_size += sizeof(AVS_ONNX_FILTER) + AVS_MEM_ALIGN_128BYTE;


	mem_tab[0].size = mem_size;
	mem_tab[0].base = NULL;
	mem_tab[0].alignment = AVS_MEM_ALIGN_128BYTE;
	return 0;
}

unsigned int AVS_Onnx_CreatMemSize(	AVS_ONNX_INFO		*input,
									AVS_MEM_TAB			mem_tab[AVS_ONNX_MEM_TAB],
									void				**handle) 
{
	int k = 0;
	AVS_ONNX_FILTER *filter = NULL;
	AVS_ONNX_BUF mem_buf;
	mem_buf.start = mem_tab[0].base;
	mem_buf.cur_pos = mem_tab[0].base;
	mem_buf.end = (void *)((QWORD)mem_buf.cur_pos + (QWORD)mem_tab[0].size);

	// pre.onnx输入内存, 完整的lib应该在pointpillars接口中
	input->onnx_pre_input.pillar_x = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf, 
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// pillar_x
	AVS_CHECK_ERROR(input->onnx_pre_input.pillar_x == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.pillar_y = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// pillar_y
	AVS_CHECK_ERROR(input->onnx_pre_input.pillar_y == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.pillar_z = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// pillar_z
	AVS_CHECK_ERROR(input->onnx_pre_input.pillar_z == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.pillar_i = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// pillar_i
	AVS_CHECK_ERROR(input->onnx_pre_input.pillar_i == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.num_voxels = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * sizeof(float));	// num_voxels
	AVS_CHECK_ERROR(input->onnx_pre_input.num_voxels == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.x_sub_shaped = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// x_sub_shaped
	AVS_CHECK_ERROR(input->onnx_pre_input.x_sub_shaped == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.y_sub_shaped = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// y_sub_shaped
	AVS_CHECK_ERROR(input->onnx_pre_input.y_sub_shaped == NULL, AVS_LIB_PTR_NULL);

	input->onnx_pre_input.mask = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM * sizeof(float));	// mask
	AVS_CHECK_ERROR(input->onnx_pre_input.mask == NULL, AVS_LIB_PTR_NULL);

	// pre.onnx输出内存, 完整的lib应该在pointpillars接口中
	input->onnx_scatter_in.pillar_feature = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_PREONNX_OUT_C * AVS_PREONNX_MAX_IN_C * sizeof(float));	// 	pillars feature
	AVS_CHECK_ERROR(input->onnx_scatter_in.pillar_feature == NULL, AVS_LIB_PTR_NULL);

	// rpn.onnx输入内存
	input->onnx_rpn_in.spatial_features= (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_C * AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * sizeof(float));	// 	spatial_features
	AVS_CHECK_ERROR(input->onnx_rpn_in.spatial_features == NULL, AVS_LIB_PTR_NULL);

	// rpn.onnx输出内存
	input->onnx_rpn_out.out_184 = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1 * sizeof(float));	// 	184
	AVS_CHECK_ERROR(input->onnx_rpn_out.out_184 == NULL, AVS_LIB_PTR_NULL);

	input->onnx_rpn_out.out_185 = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	185
	AVS_CHECK_ERROR(input->onnx_rpn_out.out_185 == NULL, AVS_LIB_PTR_NULL);

	input->onnx_rpn_out.out_187 = (float *)AVS_ONNX_COM_alloc_buffer(&mem_buf,
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2 * sizeof(float));	// 	187
	AVS_CHECK_ERROR(input->onnx_rpn_out.out_187 == NULL, AVS_LIB_PTR_NULL);

	// 算法库句柄内存
	filter = (AVS_ONNX_FILTER *)AVS_ONNX_COM_alloc_buffer(&mem_buf, sizeof(AVS_ONNX_FILTER));
	AVS_CHECK_ERROR(filter == NULL, AVS_LIB_PTR_NULL);

	*handle = (void *)filter;
	return 0;
}

unsigned int AVS_Onnx_Init(	void						*handle, 
							AVS_ONNX_INFO				*inbuf,
							int							in_buf_size) 
{
	AVS_ONNX_FILTER *filter = (AVS_ONNX_FILTER *)handle;

	// 下列初始化二选一即可 opencv dnn库支持性不太好
	// opencv dnn onnx初始化
	// dnn_onnx_init(filter->net, inbuf->onnx);

	// onnxruntime pre初始化
	onnxruntime_init(filter, inbuf->onnx);

	// onnxruntime rpn初始化
	onnxruntime_rpn_init(filter, inbuf->rpn_onnx);

	printf("onnx init done!");
	return 0;
}

// https://www.pianshen.com/article/3342263430/
// https://blog.csdn.net/wanggao_1990/article/details/86713653 dnn函数介绍
// https://docs.opencv.org/3.4/db/d30/classcv_1_1dnn_1_1Net.html#abf96c5e92de4f6cd3013a6fb900934b4
unsigned int AVS_Onnx_Forward(	void						*handle,
								AVS_ONNX_INFO				*inbuf,
								int							in_buf_size,
								AVS_ONNX_INFO				*outbuf,
								int							out_buf_size) 
{
	AVS_ONNX_FILTER *filter = (AVS_ONNX_FILTER *)handle;

	// opencv dnn 显示接口
	// dnn_onnx_show(filter->net);
	// opencv dnn 根据图像进行前向
	// dnn_onnx_image_forward(filter, inbuf->image);
	// opencv dnn 根据向量进行前向
	// dnn_onnx_vector_forward(filter->net, &inbuf->onnx_pre_input);
	// printf("opencv dnn done!\n");

	// onnxruntime 接口
	 onnxruntime_show(filter->session);
	// onnxruntime 输入生成
	float *ort_outputs = onnxruntime_foward(filter->session, inbuf->onnx_pre_input);
	// onnxruntime 结果输出
	onnxruntime_output(ort_outputs, inbuf->onnx_scatter_in);
	printf("onnxruntime pre done!\n");

	return 0;
}

unsigned int AVS_Onnx_Rpn_Forward(	void						*handle,
									AVS_ONNX_INFO				*inbuf,
									int							in_buf_size,
									AVS_ONNX_INFO				*outbuf,
									int							out_buf_size)
{
	AVS_ONNX_FILTER *filter = (AVS_ONNX_FILTER *)handle;

	// onnxruntime 接口
	onnxruntime_show(filter->rpn_session);
	// onnxruntime 输入生成
	vector<float *> ort_outputs = onnxruntime_rpn_foward(filter->rpn_session, inbuf->onnx_rpn_in);
	// onnxruntime 结果输出
	onnxruntime_rpn_output(ort_outputs, outbuf->onnx_rpn_out);
	printf("onnxruntime rpn done!\n");

	return 0;
}