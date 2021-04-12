#ifndef _POST_POINTPILLARS_H_
#define _POST_POINTPILLARS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "Avs_Base.h"
#include "Avs_Common.h"
#include "Avs_ErrorCode.h"
#include "avs_pointpillars.h"

#define POINTPILALRS_POST_MEM_TAB	1

typedef struct _POINTPILLARS_POST_INFO_ {
	// post in的数据
	AVS_PP_RPN_OUT			rpn_out;
	AVS_POST_IN				post_in;
	// post out的数据
	// AVS_POST_OUT			post_out;
}POINTPILLARS_POST_INFO;

unsigned int PointPillars_Post_GetMemSize(		POINTPILLARS_POST_INFO		*input,
												AVS_MEM_TAB					mem_tab[POINTPILALRS_POST_MEM_TAB]);

unsigned int PointPillars_Post_CreatMemSize(	POINTPILLARS_POST_INFO		*input,
												AVS_MEM_TAB					mem_tab[POINTPILALRS_POST_MEM_TAB],
												void						**handle);

unsigned int PointPillars_Post_Init(			void						*handle,
												POINTPILLARS_POST_INFO		*inbuf,
												int							in_buf_size);

unsigned int PointPillars_Post_Proc(			void						*handle,
												POINTPILLARS_POST_INFO		*inbuf,
												int							in_buf_size,
												POINTPILLARS_POST_INFO		*outbuf,
												int							out_buf_size);

#ifdef __cplusplus
}
#endif

#endif
