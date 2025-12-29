#ifndef __REGION_PARAM_PARSER_EXT_H__
#define __REGION_PARAM_PARSER_EXT_H__

#include "sample_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

void parse_region_param(char *rgnArgs);
unsigned int get_mosaic_region_num();
unsigned int get_cover_region_num();
int get_mosaic_attr_by_index(unsigned int index, hi_rgn_mosaic_chn_attr *attr);
int get_cover_attr_by_index(unsigned int index, hi_rgn_cover_chn_attr *attr);

#ifdef __cplusplus
}
#endif

#endif