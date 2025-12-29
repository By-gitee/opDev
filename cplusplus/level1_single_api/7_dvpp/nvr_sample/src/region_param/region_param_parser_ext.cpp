#include "region_param_parser_ext.h"
#include "region_param_parser.h"

void parse_region_param(char *rgnArgs)
{
    RegionParamParser::GetInstance()->ParseRgnParam(rgnArgs);
}

unsigned int get_mosaic_region_num()
{
    return RegionParamParser::GetInstance()->GetMosaicRgnNum();
}

unsigned int get_cover_region_num()
{
    return RegionParamParser::GetInstance()->GetCoverRgnNum();
}

int get_mosaic_attr_by_index(unsigned int index, hi_rgn_mosaic_chn_attr *attr)
{
    if (attr == HI_NULL) {
        return -1;
    }
    return RegionParamParser::GetInstance()->GetMosaicAttrByIndex(index, *attr);
}

int get_cover_attr_by_index(unsigned int index, hi_rgn_cover_chn_attr *attr)
{
    if (attr == HI_NULL) {
        return -1;
    }
    return RegionParamParser::GetInstance()->GetCoverAttrByIndex(index, *attr);
}