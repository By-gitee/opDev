#ifndef __REGION_PARAM_PARSER_H__
#define __REGION_PARAM_PARSER_H__

#include <vector>
#include <string>
#include "sample_comm.h"

class RegionParamParser {
public:
    static RegionParamParser* GetInstance();
    void ParseRgnParam(char *rgn_argc);
    uint32_t GetMosaicRgnNum();
    uint32_t GetCoverRgnNum();
    int32_t  GetMosaicAttrByIndex(uint32_t index, hi_rgn_mosaic_chn_attr& attr);
    int32_t GetCoverAttrByIndex(uint32_t index, hi_rgn_cover_chn_attr& attr);
private:
    void ParseRgnMosaicAttr(std::string& rgnArgsStr, uint32_t& pos, hi_rgn_mosaic_chn_attr& attr);
    uint32_t ParseRgnNum(std::string& rgnArgsStr, uint32_t& pos);
    hi_rgn_type ParseRgnType(std::string& rgnArgsStr, uint32_t& pos);
private:
    std::vector<hi_rgn_mosaic_chn_attr> mosaicAttrs_;
    std::vector<hi_rgn_cover_chn_attr> coverAttrs_;
    static RegionParamParser instance_;
};

#endif