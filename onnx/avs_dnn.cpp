#include "avs_dnn.h"

// onnx 模型初始化
void dnn_onnx_init(	dnn::Net	&net, 
					char		*onnx) {
	// onnx初始化 https://blog.csdn.net/hnsdgxylh/article/details/101904843
	net = cv::dnn::readNetFromONNX(string(onnx)); //读取网络和参数
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	//net.setPreferableTarget(DNN_TARGET_OPENCL);
	net.setPreferableTarget(DNN_TARGET_CPU);
}

// 得到所有层的id, 类型，名字， 
void dnn_onnx_show(	dnn::Net	&net) {
	auto name = net.getLayerNames();	// 获取所有层名字，不包括输入层
	for (auto &x : name) {
		auto id = net.getLayerId(x);
		// getLayerShapes() 指定输入形状和layer id返回该层的输入输出大小
		auto layer = net.getLayer(id);
		cout << " layer id: " << id << " type: " << layer->type << " name: " << layer->name << endl;
	}
}

// 根据图像对onnx进行前向
void dnn_onnx_image_forward(	AVS_ONNX_FILTER *filter,
								char			*image) {
	filter->image = imread(image);
	cv::cvtColor(filter->image, filter->image, cv::COLOR_BGR2RGB);
	Mat inputBolb = blobFromImage(filter->image, 0.00390625f, Size(32, 32), Scalar(), false, false); //将图像转化为正确输入格式
	filter->net.setInputShape("data", { 1, 1, 32, 32 });
	filter->net.setInput(inputBolb, "data");
	Mat result = filter->net.forward("out");
	Mat probMat = result.reshape(1, 1); //reshape the blob to 1x1000 matrix
	double classProb = 0.f;
	Point classNumber;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	printf("class %d prob %f\n", classNumber.x, classProb);
	printf("Opencv dnn Image ForWard Done!\n");
}

void dnn_onnx_vector_forward(	dnn::Net				&net,
								AVS_PP_PREPROCESS_OUT	*onnx_pre_input) {
	int sizes[] = { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM };
	int sizes2[] = { 1, AVS_PREONNX_MAX_IN_C };
	for (int i = 0; i < AVS_PREONNX_MAX_IN_C * AVS_PREONNX_MAX_DIM; ++i)
		onnx_pre_input->pillar_x[i] = 1.f;

	memset(onnx_pre_input->num_voxels, 1, AVS_PREONNX_MAX_IN_C * sizeof(float));
	Mat pillar_x(4, sizes, CV_32F, onnx_pre_input->pillar_x);
	Mat pillar_y(4, sizes, CV_32F, onnx_pre_input->pillar_y);
	Mat pillar_z(4, sizes, CV_32F, onnx_pre_input->pillar_z);
	Mat pillar_i(4, sizes, CV_32F, onnx_pre_input->pillar_i);
	Mat num_voxels(2, sizes2, CV_32F, onnx_pre_input->num_voxels);
	Mat x_sub_shaped(4, sizes, CV_32F, onnx_pre_input->x_sub_shaped);
	Mat y_sub_shaped(4, sizes, CV_32F, onnx_pre_input->y_sub_shaped);
	Mat mask(4, sizes, CV_32F, onnx_pre_input->mask);

	net.setInputShape("pillar_x", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_y", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_z", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("pillar_i", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("num_points_per_pillar", { 1, AVS_PREONNX_MAX_IN_C });
	net.setInputShape("x_sub_shaped", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("y_sub_shaped", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });
	net.setInputShape("mask", { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM });

	net.setInput(pillar_x, "pillar_x");
	net.setInput(pillar_y, "pillar_y");
	net.setInput(pillar_z, "pillar_z");
	net.setInput(pillar_i, "pillar_i");
	net.setInput(num_voxels, "num_points_per_pillar");	// dnn的输入是否一定是nchw?
	net.setInput(x_sub_shaped, "x_sub_shaped");
	net.setInput(y_sub_shaped, "y_sub_shaped");
	net.setInput(mask, "mask");
	Mat result = net.forward("163");	// 该onnx结构中的scale不支持
	printf("Opencv dnn ForWard Done!\n");
}