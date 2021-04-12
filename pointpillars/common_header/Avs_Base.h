#ifndef _AVS_BASE_H_
#define _AVS_BASE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define AVS_MAX_MEM_NUM			4
#define AVS_MEM_ALIGN_128BYTE	128
#define AVS_PI					3.141592653

#define QWORD unsigned long long

typedef struct _AVS_POINT_I_ {
	int x;
	int y;
	int z;
}AVS_POINT_I;

typedef struct _AVS_POINT_F_ {
	float x;
	float y;
	float z;
}AVS_POINT_F;

typedef struct _AVS_MEM_TAB_ {
	unsigned int size;
	unsigned int alignment;
	void		 *base;
}AVS_MEM_TAB;

typedef struct _AVS_MEM_ {
	int mem_tab_num;
	AVS_MEM_TAB mem_tab;
	//AVS_MEM_TAB mem_tab[AVS_MAX_MEM_NUM];
}AVS_MEM;

#ifdef __cplusplus
}
#endif

#endif