#include <iostream>
#include "region_param_parser.h"

const std::string pattern(":");

RegionParamParser RegionParamParser::instance_;

void RegionParamParser::ParseRgnMosaicAttr(std::string& rgnArgsStr, uint32_t& pos, hi_rgn_mosaic_chn_attr& attr)
{
    uint32_t patternPos;
    patternPos = rgnArgsStr.find(pattern, pos);
    attr.rect.x = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;

    patternPos = rgnArgsStr.find(pattern, pos);
    attr.rect.y = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;

    patternPos = rgnArgsStr.find(pattern, pos);
    attr.rect.width = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;

    patternPos = rgnArgsStr.find(pattern, pos);
    attr.rect.height = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;

    patternPos = rgnArgsStr.find(pattern, pos);
    attr.blk_size = (hi_blk_size)atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;

    patternPos = rgnArgsStr.find(pattern, pos);
    attr.layer = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;
    std::cout << "attr.rect.x:" <<  attr.rect.x << std::endl;
    std::cout << "attr.rect.y:" <<  attr.rect.y << std::endl;
    std::cout << "attr.rect.width:" <<  attr.rect.width << std::endl;
    std::cout << "attr.rect.height:" <<  attr.rect.height << std::endl;
    std::cout << "attr.blk_size:" <<  attr.blk_size << std::endl;
    std::cout << "attr.layer:" <<  attr.layer << std::endl;
}

uint32_t RegionParamParser::ParseRgnNum(std::string& rgnArgsStr, uint32_t& pos)
{
    uint32_t patternPos;
    patternPos = rgnArgsStr.find(pattern, pos);
    uint32_t rgn_num = atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;
    std::cout << "rgn_num:" <<  rgn_num << std::endl;
    return rgn_num;
}

hi_rgn_type RegionParamParser::ParseRgnType(std::string& rgnArgsStr, uint32_t& pos)
{
    uint32_t patternPos;
    patternPos = rgnArgsStr.find(pattern, pos);
    hi_rgn_type rgn_type = (hi_rgn_type)atoi(rgnArgsStr.substr(pos, patternPos).c_str());
    pos = patternPos + 1;
    std::cout << "rgn_type: [" <<  rgn_type << "]" << std::endl;
    return rgn_type;
}

RegionParamParser* RegionParamParser::GetInstance()
{
    return &instance_;
}

/*
    RegionParam
    rgnNum:rgnType:mosaicRectX:mosaicRectY:mosaicRectWidth:mosaicRectHeight:mosaicBlkSize:mosaicLayer....
*/
void RegionParamParser::ParseRgnParam(char *rgnArgs)
{
    if (rgnArgs == HI_NULL) {
        return;
    }
    std::string rgnArgsStr(rgnArgs);

    uint32_t pos = 0;
    uint32_t rgnNum = ParseRgnNum(rgnArgsStr, pos);
    hi_rgn_mosaic_chn_attr mosaicAttr;
    hi_rgn_cover_chn_attr coverAttr;
    hi_rgn_type rgnType;
    for (uint32_t i = 0; i < rgnNum; i++) {
        rgnType = ParseRgnType(rgnArgsStr, pos);
        switch (rgnType) {
            case HI_RGN_MOSAIC:
                ParseRgnMosaicAttr(rgnArgsStr, pos, mosaicAttr);
                mosaicAttrs_.push_back(mosaicAttr);
                continue;
            default:
                return;
        }
    }
}

uint32_t RegionParamParser::GetMosaicRgnNum()
{
    return mosaicAttrs_.size();
}

uint32_t RegionParamParser::GetCoverRgnNum()
{
    return coverAttrs_.size();
}

int32_t RegionParamParser::GetMosaicAttrByIndex(uint32_t index, hi_rgn_mosaic_chn_attr& attr)
{
    if (index >= mosaicAttrs_.size()) {
        return -1;
    }
    attr = mosaicAttrs_[index];
    return 0;
}

int32_t RegionParamParser::GetCoverAttrByIndex(uint32_t index, hi_rgn_cover_chn_attr& attr)
{
    if (index >= coverAttrs_.size()) {
        return -1;
    }
    attr = coverAttrs_[index];
    return 0;
}