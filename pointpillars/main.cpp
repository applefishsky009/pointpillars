#include "test.hpp"

// https://github.com/SmallMunich/nutonomy_pointpillars
// C implement
#define TEST_PRE	0	// pre.onnx测试
#define TEST_RPN	0	// rpn.onnx测试
#define TEST_POST	1	// post测试

int main() {
#if TEST_PRE || TEST_RPN
	// 变量初始化
	unsigned int	ret = 0;
	AVS_ONNX_INFO	onnx_info;
	AVS_MEM_TAB		onnx_mem_tab[AVS_ONNX_MEM_TAB];
	// 内存计算
	ret = AVS_Onnx_GetMemSize(&onnx_info, onnx_mem_tab);
	AVS_CHECK_ERROR(ret != AVS_LIB_OK, ret);
	// 内存申请
	AVS_alloc_mem_tab(onnx_mem_tab, AVS_ONNX_MEM_TAB);
	memset(onnx_mem_tab->base, 0, onnx_mem_tab->size);
	// 内存分配
	void *filter_onnx	= nullptr;
	ret = AVS_Onnx_CreatMemSize(&onnx_info, onnx_mem_tab, &filter_onnx);
	AVS_CHECK_ERROR(ret != AVS_LIB_OK, ret);
	// 算法句柄初始化
	onnx_info.onnx		= "../pfe_1.7.0.onnx";
	onnx_info.rpn_onnx	= "../rpn_1.7.0.onnx";
	onnx_info.image		= "../verify_undistort.png";
	ret = AVS_Onnx_Init(filter_onnx, &onnx_info, sizeof(AVS_ONNX_INFO));
	AVS_CHECK_ERROR(ret != AVS_LIB_OK, ret);
#endif
#if TEST_PRE
	// pre数据读入
	string pillar_x			= "../txts/0.pillar_x.txt";
	string pillar_y			= "../txts/1.pillar_y.txt";
	string pillar_z			= "../txts/2.pillar_z.txt";
	string pillar_i			= "../txts/3.pillar_i.txt";
	string num_voxels		= "../txts/4.num_voxels.txt";
	string x_sub_shaped		= "../txts/5.x_sub_shaped.txt";
	string y_sub_shaped		= "../txts/6.y_sub_shaped.txt";
	string mask				= "../txts/7.mask.txt";
	string pillar_feature	= "../txts/8.pillar_feature_win64.txt";
	get_onnx_input(pillar_x, onnx_info.onnx_pre_input.pillar_x);
	get_onnx_input(pillar_y, onnx_info.onnx_pre_input.pillar_y);
	get_onnx_input(pillar_z, onnx_info.onnx_pre_input.pillar_z);
	get_onnx_input(pillar_i, onnx_info.onnx_pre_input.pillar_i);
	get_onnx_input(num_voxels, onnx_info.onnx_pre_input.num_voxels);
	get_onnx_input(x_sub_shaped, onnx_info.onnx_pre_input.x_sub_shaped);
	get_onnx_input(y_sub_shaped, onnx_info.onnx_pre_input.y_sub_shaped);
	get_onnx_input(mask, onnx_info.onnx_pre_input.mask);
	// pre前向推理 遇到一个问题, 
	// pytorch输入12000，输出12000, onnx输入5000，输出5000，输入的7000置0，数据紧密排列，但输出的7000不是0，需要加上网络的bias!
	ret = AVS_Onnx_Forward(filter_onnx, 
		&onnx_info, sizeof(AVS_ONNX_INFO),
		&onnx_info, sizeof(AVS_ONNX_INFO));
	AVS_CHECK_ERROR(ret != AVS_LIB_OK, ret);
	// pre结果输出
	write_onnx_out(onnx_info.onnx_scatter_in.pillar_feature, pillar_feature, 
		AVS_PREONNX_OUT_C * AVS_PREONNX_MAX_IN_C);
#endif
#if TEST_RPN
	// rpn数据读入
	string rpn_in	= "../txts/9.rpn_in.txt";
	string rpn_out1 = "../txts/10.rpn_out1_win64.txt";
	string rpn_out2 = "../txts/11.rpn_out2_win64.txt";
	string rpn_out3	= "../txts/12.rpn_out3_win64.txt";
	get_onnx_input(rpn_in, onnx_info.onnx_rpn_in.spatial_features);

	// rpn前向推理
	printf("rpn proc...\n");
	AVS_Onnx_Rpn_Forward(filter_onnx,
		&onnx_info, sizeof(AVS_ONNX_INFO),
		&onnx_info, sizeof(AVS_ONNX_INFO));
	AVS_CHECK_ERROR(ret != AVS_LIB_OK, ret);
	printf("end rpn proc...\n");
	// rpn结果输出
	write_onnx_out(onnx_info.onnx_rpn_out.out_184, rpn_out1, 
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1);
	write_onnx_out(onnx_info.onnx_rpn_out.out_185, rpn_out2, 
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2);
	write_onnx_out(onnx_info.onnx_rpn_out.out_187, rpn_out3, 
		AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2);
#endif

#if TEST_POST
	unsigned int			ans		= 0;
	POINTPILLARS_POST_INFO	post_in;
	AVS_MEM_TAB				post_mem_tab[POINTPILALRS_POST_MEM_TAB];
	// 内存计算
	ans = PointPillars_Post_GetMemSize(&post_in, post_mem_tab);
	AVS_CHECK_ERROR(ans != AVS_LIB_OK, ans);
	// 内存申请
	AVS_alloc_mem_tab(post_mem_tab, POINTPILALRS_POST_MEM_TAB);
	memset(post_mem_tab->base, 0, post_mem_tab->size);
	// 内存分配
	void *filter_post = nullptr;
	ans = PointPillars_Post_CreatMemSize(&post_in, post_mem_tab, &filter_post);
	AVS_CHECK_ERROR(ans != AVS_LIB_OK, ans);
	// 算法句柄初始化
	ans = PointPillars_Post_Init(filter_post, &post_in, sizeof(POINTPILLARS_POST_INFO));
	AVS_CHECK_ERROR(ans != AVS_LIB_OK, ans);
	// post数据读入
	float post_rect[4][4] = {
		{0.99992388,  0.00983776, -0.00744505,  0.00000000},
		{-0.00986980,  0.99994212, -0.00427846,  0.00000000},
		{0.00740253,  0.00435161,  0.99996310,  0.00000000},
		{0.00000000,  0.00000000,  0.00000000,  1.00000000} };
	memcpy(post_in.post_in.post_rect, post_rect, 4 * 4 * sizeof(float));	// rect
	float trv2c[4][4] = {
		{7.53374491e-03, -9.99971390e-01, -6.16602018e-04, -4.06976603e-03},
		{1.48024904e-02,  7.28073297e-04, -9.99890208e-01, -7.63161778e-02},
		{9.99862075e-01,  7.52379000e-03,  1.48075502e-02, -2.71780610e-01},
		{0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00} };
	memcpy(post_in.post_in.trv2c, trv2c, 4 * 4 * sizeof(float));	// trv2c
	float p2[4][4] = {
		{7.21537720e+02, 0.00000000e+00, 6.09559326e+02, 4.48572807e+01},
		{0.00000000e+00, 7.21537720e+02, 1.72854004e+02, 2.16379106e-01},
		{0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 2.74588400e-03},
		{0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00} };
	memcpy(post_in.post_in.p2, p2, 4 * 4 * sizeof(float));	// p2
	string out_1		= "../txts/post_preds_dict0.txt";
	string out_2		= "../txts/post_preds_dict1.txt";
	string out_3		= "../txts/post_preds_dict2.txt";
	string anchor		= "../txts/post_anchors_o.txt";
	string anchor_mask	= "../txts/post_anchors_mask_o.txt";
	get_onnx_input(out_1, post_in.rpn_out.out_184);
	get_onnx_input(out_2, post_in.rpn_out.out_185);
	get_onnx_input(out_3, post_in.rpn_out.out_187);
	get_onnx_input(anchor, post_in.post_in.anchors);
	get_onnx_input(anchor_mask, post_in.post_in.anchor_mask);

	// post 后处理
	ans = PointPillars_Post_Proc(filter_post,
		&post_in, sizeof(POINTPILLARS_POST_INFO),
		&post_in, sizeof(POINTPILLARS_POST_INFO));
	AVS_CHECK_ERROR(ans != AVS_LIB_OK, ans);
	// post 结果输出

#endif
	return 0;
}
