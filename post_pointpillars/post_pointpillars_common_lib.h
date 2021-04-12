#ifndef _POST_POINTPILLARS_COMMON_LIB_H_
#define _POST_POINTPILLARS_COMMON_LIB_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "Avs_Base.h"
#include "Avs_Common.h"

#define AVS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define AVS_MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct _AVS_POST_POINTPILLARS_BUF_ {
	void *start;
	void *end;
	void *cur_pos;
}AVS_POST_POINTPILLARS_BUF;

void *AVS_POST_COM_alloc_buffer(AVS_POST_POINTPILLARS_BUF	*avs_buf,
								int							size);

#ifdef __cplusplus
}
#endif

#endif
