#include "test.hpp"

void *alloc_memory(unsigned int size, unsigned int aligment) {
	void *data;
	data = (char *)_aligned_malloc(size, aligment);
	return data;
}

void AVS_alloc_mem_tab(AVS_MEM_TAB *mem_tab, int num) {
	for (int i = 0; i < num; ++i) {
		if (mem_tab[i].size > 0) {
			mem_tab[i].base = alloc_memory(mem_tab[i].size, mem_tab[i].alignment);
		}
		else {
			mem_tab[i].base = NULL;
		}
		printf("tab %d memsize: %f M\n", i, mem_tab[i].size / 1024.0 / 1024.0);
	}
}

// 行信息转为float
void string2float(string &radar_point, vector<float> &point_info) {
	radar_point.push_back(' ');
	size_t pos = radar_point.find_first_of(' ');
	while (pos != string::npos) {
		string string_part = radar_point.substr(0, pos);
		float float_part = atof(string_part.c_str());
		point_info.push_back(float_part);
		radar_point = radar_point.substr(pos + 1);
		pos = radar_point.find_first_of(' ');
	}
}
// 从文件读入激光雷达信息
void get_onnx_input(string &input_txt, float *data) {

	int i = 0, num_cnt = 0, row_cnt = 0;
	ifstream fin(input_txt);
	string lidar_point;
	while (getline(fin, lidar_point)) {
		vector<float> point_info;
		string2float(lidar_point, point_info);	// 将一行字符串转换为vector
		// 将vector数据写入lib输入格式
		++row_cnt;
		for (i = 0; i < point_info.size(); ++i)
			data[num_cnt++] = point_info[i];
	}
	printf("read %s info done, input_txt have %d x %d dims!\n", input_txt.c_str(), row_cnt, num_cnt / row_cnt);
}

void write_onnx_out(float *pillar_feature, string &pillar_ftxt, int len) {
	int i = 0;
	ofstream out(pillar_ftxt, ios::out);
	out.precision(18);
	//out.flags(ios::left | ios::fixed);
	out.flags(ios::left | ios::scientific);
	out.fill('0');
	for (i = 0; i < len; ++i) {
		out << pillar_feature[i] << endl;
	}
	//out.width(8);
	out.close();
	printf("pre_onnx result wirte done!\n");
}