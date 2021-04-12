#include <sstream>
#include <functional>   // std::minus
#include <numeric>      // std::accumulate
#include "avs_onnxruntime.h"

// onnx pre模型初始化
void onnxruntime_init(	AVS_ONNX_FILTER *filter,
						char			*onnx) {

	// initialize  enviroment...one enviroment per process
	filter->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);	// set ops
	// 注意头目录的包含顺序,因为cpu和gpu的库名字相同,先包含cpu会导致gpu接口无法使用
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);	// gpu id
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	printf("Using Onnxruntime C++ API\n");
	// char 转wchar_t
	wstringstream onnx_model;
	onnx_model << onnx;
	// Ort::Session session(env, onnx_model.str().c_str(), session_options);
	filter->session = Ort::Session(filter->env, onnx_model.str().c_str(), session_options);
	printf("onnxruntime pre session init done!\n");
}

// onnx rpn模型初始化
void onnxruntime_rpn_init(	AVS_ONNX_FILTER *filter,
							char			*rpn_onnx) {

	// initialize  enviroment...one enviroment per process
	filter->rpn_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);	// set ops
	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);	// gpu id
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	printf("Using Onnxruntime C++ API\n");
	// char 转wchar_t
	wstringstream rpn_onnx_model;
	rpn_onnx_model << rpn_onnx;
	// Ort::Session session(env, onnx_model.str().c_str(), session_options);
	filter->rpn_session = Ort::Session(filter->rpn_env, rpn_onnx_model.str().c_str(), session_options);
	printf("onnxruntime rpn session init done!\n");
}

void onnxruntime_show(	Ort::Session	&session) {
	// 获取输入节点个数
	Ort::AllocatorWithDefaultOptions allocator;
	size_t num_input_nodes = session.GetInputCount();	
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;
	printf("Number of inputs = %zu\n", num_input_nodes);
	// 遍历输入节点
	for (int i = 0; i < num_input_nodes; i++) {
		// 得到输入节点名字
		char* input_name = session.GetInputName(i, allocator);
		printf("Input %d : name=%s\n", i, input_name);
		input_node_names[i] = input_name;

		// 得到输入节点类型
		Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		ONNXTensorElementDataType type = tensor_info.GetElementType();
		printf("Input %d : type=%d\n", i, type);

		// 获取输入节点维度
		input_node_dims = tensor_info.GetShape();
		printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
		for (int j = 0; j < input_node_dims.size(); j++)
			printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
	}
}

Ort::Value generate_tensor(	vector<int64_t> dims, 
							float *data,
							std::vector<float> &input_vector) {
	int i = 0;
	size_t dims_size = std::accumulate(dims.begin(), dims.end(), 1, multiplies<int64_t>());	// 总大小
	if (data) {	// fill values
		for (i = 0; i < dims_size; ++i) {
			input_vector[i] = data[i];
		}
	}
	Ort::MemoryInfo input_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(input_memory_info,
		input_vector.data(), dims_size, dims.data(), dims.size());
	assert(input_tensor.IsTensor());
	return input_tensor;
}

float* onnxruntime_foward(	Ort::Session			&session,
							AVS_PP_PREPROCESS_OUT	&onnx_pre_input) {
	// 参数初始化
	std::vector<Ort::Value> ort_inputs;
	std::vector<int64_t> input_pillar_dims = { 1, 1, AVS_PREONNX_MAX_IN_C, AVS_PREONNX_MAX_DIM };
	std::vector<int64_t> input_voxel_dims = { 1, AVS_PREONNX_MAX_IN_C };
	size_t pillar_dims_size = std::accumulate(input_pillar_dims.begin(), input_pillar_dims.end(), 1, multiplies<int64_t>());	// 总大小
	size_t voxel_dims_size = std::accumulate(input_voxel_dims.begin(), input_voxel_dims.end(), 1, multiplies<int64_t>());	// 总大小

	// 设置输入向量 * 8 全0初始化
	std::vector<float> input_pillar_x_values(pillar_dims_size, 0);	// vector变量作用域必须和tensor保持一致
	Ort::Value pillar_x_tensor = generate_tensor(input_pillar_dims, 
		onnx_pre_input.pillar_x, input_pillar_x_values);
	ort_inputs.push_back(std::move(pillar_x_tensor));

	std::vector<float> input_pillar_y_values(pillar_dims_size, 0);
	Ort::Value pillar_y_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.pillar_y, input_pillar_y_values);
	ort_inputs.push_back(std::move(pillar_y_tensor));

	std::vector<float> input_pillar_z_values(pillar_dims_size, 0);
	Ort::Value pillar_z_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.pillar_z, input_pillar_z_values);
	ort_inputs.push_back(std::move(pillar_z_tensor));

	std::vector<float> input_pillar_i_values(pillar_dims_size, 0);
	Ort::Value pillar_i_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.pillar_i, input_pillar_i_values);
	ort_inputs.push_back(std::move(pillar_i_tensor));

	std::vector<float> input_voxels_values(voxel_dims_size, 0);
	Ort::Value num_voxels_tensor = generate_tensor(input_voxel_dims,
		onnx_pre_input.num_voxels, input_voxels_values);
	ort_inputs.push_back(std::move(num_voxels_tensor));

	std::vector<float> input_x_sub_shaped_values(pillar_dims_size, 0);
	Ort::Value x_sub_shaped_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.x_sub_shaped, input_x_sub_shaped_values);
	ort_inputs.push_back(std::move(x_sub_shaped_tensor));

	std::vector<float> input_y_sub_shaped_values(pillar_dims_size, 0);
	Ort::Value y_sub_shaped_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.y_sub_shaped, input_y_sub_shaped_values);
	ort_inputs.push_back(std::move(y_sub_shaped_tensor));

	std::vector<float> input_mask_values(pillar_dims_size, 0);
	Ort::Value mask_tensor = generate_tensor(input_pillar_dims,
		onnx_pre_input.mask, input_mask_values);
	ort_inputs.push_back(std::move(mask_tensor));

	// 设置输入输出层名字
	std::vector<const char*> input_names = { 
		"pillar_x", "pillar_y", "pillar_z", "pillar_i",
		"num_points_per_pillar", "x_sub_shaped", "y_sub_shaped", "mask" };
	std::vector<const char*> output_names = { "174" };

	// 前向
	auto ort_outputs = session.Run(Ort::RunOptions{ nullptr },
		input_names.data(), ort_inputs.data(), input_names.size(),
		output_names.data(), output_names.size());
	float* pillar_feature = ort_outputs[0].GetTensorMutableData<float>();

	// 前向的另一种形式的写法
	//std::vector<int64_t> output_pillar_dims = { 1, AVS_PREONNX_OUT_C, AVS_PREONNX_MAX_IN_C, 1 };
	//size_t out_dims_size = std::accumulate(input_pillar_dims.begin(), input_pillar_dims.end(), 1, multiplies<int64_t>());	// 总大小
	//std::vector<float> output_values(out_dims_size, 1); 
	//Ort::Value output_pillar_tensor = generate_tensor(output_pillar_dims, nullptr, output_values);

	//session.Run(Ort::RunOptions{ nullptr },
	//	input_names.data(), ort_inputs.data(), ort_inputs.size(), 
	//	output_names.data(), &output_pillar_tensor, output_names.size());
	//float* pillar_feature = output_pillar_tensor.GetTensorMutableData<float>();

	printf("Pre ForWard Done!\n");
	return pillar_feature;
}

void onnxruntime_output(	float					*ort_outputs,
							AVS_PP_SCATTER_IN		&onnx_scatter_in) {
	int i = 0;
	for (i = 0; i < AVS_PREONNX_OUT_C * AVS_PREONNX_MAX_IN_C; ++i) {
		onnx_scatter_in.pillar_feature[i] = ort_outputs[i];
	}
}

vector<float *> onnxruntime_rpn_foward(	Ort::Session			&session,
										AVS_PP_RPN_IN			&onnx_rpn_input) {
	// 参数初始化
	vector<float *> outputs;
	std::vector<Ort::Value> ort_inputs;
	std::vector<int64_t> input_rpn_dims = { 1, AVS_RPNONNX_IN_C, AVS_RPNONNX_IN_H, AVS_RPNONNX_IN_W };
	size_t rpn_dims_size = std::accumulate(input_rpn_dims.begin(), input_rpn_dims.end(), 1, multiplies<int64_t>());	// 总大小

	// 设置输入向量 * 1
	std::vector<float> input_rpn_values(rpn_dims_size, 0);	// vector变量作用域必须和tensor保持一致
	Ort::Value pillar_x_tensor = generate_tensor(input_rpn_dims,
		onnx_rpn_input.spatial_features, input_rpn_values);
	ort_inputs.push_back(std::move(pillar_x_tensor));

	// 设置输入输出层名字
	std::vector<const char*> input_names = { "input.1" };
	std::vector<const char*> output_names = { "184", "185", "187" };

	// 前向
	auto ort_outputs = session.Run(Ort::RunOptions{ nullptr },
		input_names.data(), ort_inputs.data(), input_names.size(),
		output_names.data(), output_names.size());
	float* feature_184 = ort_outputs[0].GetTensorMutableData<float>();	// 496 x 432 x 28
	// auto xxx = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	float* feature_185 = ort_outputs[1].GetTensorMutableData<float>();	// 496 x 432 x 8
	float* feature_187 = ort_outputs[2].GetTensorMutableData<float>();	// 496 x 432 x 8
	outputs.push_back(feature_184);
	outputs.push_back(feature_185);
	outputs.push_back(feature_187);

	printf("Rpn ForWard Done!\n");
	return outputs;
}

// onnx pre模型输出
void onnxruntime_rpn_output(	vector<float *>			&ort_outputs,
								AVS_PP_RPN_OUT			&onnx_rpn_out){
	int i = 0;
	for (i = 0; i < AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_1; ++i) {
		onnx_rpn_out.out_184[i] = ort_outputs[0][i];
	}
	for (i = 0; i < AVS_RPNONNX_IN_H * AVS_RPNONNX_IN_W * AVS_RPNONNX_OUT_2; ++i) {
		onnx_rpn_out.out_185[i] = ort_outputs[1][i];
		onnx_rpn_out.out_187[i] = ort_outputs[2][i];
	}
}
