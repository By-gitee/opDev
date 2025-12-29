#ifndef FASTGELU_TILING_H
#define FASTGELU_TILING_H
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_CORE_NUM = 8;
constexpr uint32_t UB_SIZE = 8192;

struct SplitTilingData
{
    uint32_t totalLength;
    uint32_t tileNumMean;
    uint32_t tileNumEnd;
    uint32_t tileLengthMean;
    uint32_t tileLengthEnd;
    uint32_t blockLengthMean;
    uint32_t blockLengthEnd;
};

template <typename T>
void compute_splitTiling_v2(size_t totalByte, uint32_t coreNum, SplitTilingData* splitTD) {
    assert(coreNum >= 1 && "core num at least use one");
    uint32_t pad32 = BLOCK_SIZE;
    uint32_t padMax = UB_SIZE / sizeof(T);
    uint32_t totalLength = totalByte / sizeof(T); // 将字节量转换为元素个数
    splitTD->totalLength = totalLength;
    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < pad32) {
        splitTD->blockLengthMean = pad32;
        splitTD->blockLengthEnd = totalLength;
        splitTD->tileNumMean = 1;
        splitTD->tileNumEnd = 1;
        splitTD->tileLengthMean = totalLength;
        splitTD->tileLengthEnd = totalLength;
        return;
    }
    // 总数据至少比32B大时
    uint32_t realTotalLength = totalLength % (pad32 * coreNum) ? // 补足totalLength到32B倍核心数的整数倍
        ((totalLength / (pad32 * coreNum)) + 1) * (pad32 * coreNum) : totalLength;
    uint32_t maxBlockLength = realTotalLength / coreNum;
    if (realTotalLength - totalLength > maxBlockLength) {
        maxBlockLength = totalLength / coreNum;
    }
    // std::cout << realTotalLength << "," << realTotalLength - totalLength << "," << maxBlockLength << std::endl;
    if (maxBlockLength > padMax) { // maxBlockLength大于padMax时对maxBlockLength进行判定
        uint32_t padTemp = 0;
        for (int32_t i = padMax / 2; i <= padMax; i += pad32) {
            padTemp = maxBlockLength % i == 0 ? i : padTemp;
        }
        if (padTemp) { // 如果maxBlockLength可以被PadTemp整除，那么padTemp就是tilelength
            splitTD->blockLengthMean = maxBlockLength;
            splitTD->blockLengthEnd = totalLength - splitTD->blockLengthMean * (coreNum - 1);
            splitTD->tileNumMean = splitTD->blockLengthMean / padTemp;
            splitTD->tileNumEnd = splitTD->tileNumMean;
            splitTD->tileLengthMean = padTemp;
            splitTD->tileLengthEnd = splitTD->blockLengthEnd - padTemp * (splitTD->tileNumEnd - 1);
        }
        else { // 如果maxBlockLength不能被PadTemp整除，那么padMax就是tilelength
            splitTD->blockLengthMean = maxBlockLength - maxBlockLength % padMax;
            splitTD->blockLengthEnd = totalLength - splitTD->blockLengthMean * (coreNum - 1);
            splitTD->tileNumMean = splitTD->blockLengthMean / padMax;
            splitTD->tileNumEnd = splitTD->blockLengthEnd % padMax ? splitTD->blockLengthEnd / padMax + 1 : (splitTD->blockLengthEnd / padMax); // 计算最后一个核心会不会多一个尾数块
            if (padMax >= splitTD->blockLengthEnd) {
                splitTD->tileNumEnd = 1;
            }
            splitTD->tileLengthMean = padMax;
            splitTD->tileLengthEnd = splitTD->blockLengthEnd - padMax * (splitTD->tileNumEnd - 1); // 计算最后一个核心的尾数块长度
        }
    }
    else { // maxBlockLength小于padMax时直接取maxBlockLength中的最大Pad32倍数
        if (maxBlockLength >= pad32) { // maxBlockLength大于pad32时
            splitTD->blockLengthMean = maxBlockLength - maxBlockLength % pad32;
            splitTD->blockLengthEnd = totalLength - splitTD->blockLengthMean * (coreNum - 1);
            splitTD->tileNumMean = 1; // 只有一个tileNum
            splitTD->tileNumEnd = splitTD->blockLengthEnd % pad32 ? splitTD->blockLengthEnd / splitTD->blockLengthMean + 1 : splitTD->blockLengthEnd / splitTD->blockLengthMean; // 如果尾块不能32B对齐则多分配一个尾块
            if (splitTD->blockLengthMean >= splitTD->blockLengthEnd) {
                splitTD->tileNumEnd = 1;
            }
            splitTD->tileLengthMean = splitTD->blockLengthMean;
            splitTD->tileLengthEnd = splitTD->blockLengthEnd - splitTD->tileLengthMean * (splitTD->tileNumEnd - 1); // 将尾数彻底分给最后一个核心的最后一个tile

        }
        else { // maxBlockLength小于pad32时，前面的block优先分配32B数据
            splitTD->blockLengthMean = pad32;
            splitTD->blockLengthEnd = totalLength - pad32 * (coreNum - 1);
            splitTD->tileNumMean = 1;
            splitTD->tileNumEnd = 1;
            splitTD->tileLengthMean = pad32;
            splitTD->tileLengthEnd = splitTD->blockLengthEnd;
        }
    }
}


template <typename T>
void compute_splitTiling_v1(size_t totalByte, uint32_t coreNum, SplitTilingData* splitTD) {
    uint32_t pad32 = 32 / sizeof(T);
    uint32_t padMax = 8192 / sizeof(T);
    uint32_t totalLength = totalByte / sizeof(T); // 将字节量转换为元素个数
    splitTD->totalLength = totalLength;
    // 如果总数据比32B还小，直接当尾数处理
    if (totalLength < pad32) {
        splitTD->blockLengthMean = pad32;
        splitTD->blockLengthEnd = pad32;
        splitTD->tileNumMean = 1;
        splitTD->tileNumEnd = 1;
        splitTD->tileLengthMean = totalLength;
        splitTD->tileLengthEnd = totalLength;
    }
    else { // 总数据至少比32B大时
        if (coreNum > 1) { // 如果核心数量大于1，那么需要区分每个核心是否能分到32B倍数的数据
            if (totalLength % (pad32 * coreNum) == 0) { // 如果平均分个每个核心的数据都是32B倍数
                splitTD->blockLengthMean = totalLength / coreNum; // 那么直接平均分配数据给每个核心
                splitTD->blockLengthEnd = totalLength / coreNum; // 最后一个核心的数据和其他核心相同
            }
            else { // 如果平均分个每个核心的数据不是32B倍数，那么前coreNum-1个核心分配32B倍数的数据，尾数分给最后一个核心
                uint32_t realTotalLength = totalLength % (pad32 * coreNum) ? // 补足totalLength到256B倍数且是核心数的整数倍
                    ((totalLength / (pad32 * coreNum)) + 1) * (pad32 * coreNum) : totalLength;
                splitTD->blockLengthMean = realTotalLength / coreNum; // 补足的数据直接分配到前core-1个核心
                if (totalLength < splitTD->blockLengthMean * (coreNum - 1)) {
                    uint32_t padNow1 = 0;
                    for (int32_t tp1 = pad32; tp1 < splitTD->blockLengthMean; tp1 += pad32) {
                        if (realTotalLength % tp1 == 0 && (realTotalLength / tp1 >= MAX_CORE_NUM)) {
                            padNow1 = tp1;
                        }
                    }
                    splitTD->blockLengthMean = padNow1;
                    splitTD->blockLengthEnd = totalLength - splitTD->blockLengthMean * (coreNum - 1);
                }
                else {
                    splitTD->blockLengthEnd = totalLength - splitTD->blockLengthMean * (coreNum - 1); //剩下的尾数最后一个核心处理
                }
            }
        }
        else { // 如果核心数量等于1，那么全分配个唯一核心即可
            splitTD->blockLengthMean = totalLength;
            splitTD->blockLengthEnd = totalLength;
        }

        if (splitTD->blockLengthMean <= padMax) { // 如果blockLength后的数据少于一次矢量计算的最大数据量
            splitTD->tileNumMean = 1; // 那么只需要分1个tile
            splitTD->tileNumEnd = (splitTD->blockLengthEnd % pad32 != 0 && splitTD->blockLengthMean < splitTD->blockLengthEnd) ? 2 : 1; // 如果尾块不能32B对齐且Mean比End小才分配两个tile
            splitTD->tileLengthMean = splitTD->blockLengthMean;
            splitTD->tileLengthEnd = splitTD->blockLengthEnd - splitTD->tileLengthMean * (splitTD->tileNumEnd - 1); // 将尾数彻底分给最后一个核心的最后一个tile
        }        
else { // 如果blockLength后的数据超过了一次矢量计算的最大数据量
            if (splitTD->blockLengthMean % padMax == 0) { // 如果blockLengthMean/BUFFER_NUM是256B的倍数
                splitTD->tileNumMean = splitTD->blockLengthMean / padMax;
                splitTD->tileNumEnd = splitTD->blockLengthEnd % padMax ? splitTD->blockLengthEnd / padMax + 1 : (splitTD->blockLengthEnd / padMax); // 计算最后一个核心会不会多一个尾数块
                if (padMax >= splitTD->blockLengthEnd) {
                    splitTD->tileNumEnd = 1;
                }
                splitTD->tileLengthMean = padMax;
                splitTD->tileLengthEnd = splitTD->blockLengthEnd % padMax ? splitTD->blockLengthEnd - padMax * (splitTD->tileNumEnd - 1) : padMax; // 计算最后一个核心的尾数块长度
            }
            else { // 如果blockLengthMean/BUFFER_NUM不是256B的倍数, 那么查找在pad32和padMax之间的最大倍数
                uint32_t padNow = 0;
                for (int32_t tp = pad32;tp < padMax;tp += pad32) {
                    if (splitTD->blockLengthMean % tp == 0) {
                        padNow = tp;
                    }
                }
                splitTD->tileNumMean = splitTD->blockLengthMean / padNow;
                splitTD->tileNumEnd = splitTD->blockLengthEnd % padNow ? splitTD->blockLengthEnd + 1 / padNow : splitTD->blockLengthEnd / padNow;
                splitTD->tileLengthMean = splitTD->blockLengthMean / splitTD->tileNumMean; // 一定可以被blockLengthMean整除，因为blockLengthMean是pad32*coreNum整数倍
                splitTD->tileLengthEnd = splitTD->blockLengthEnd - splitTD->tileLengthMean * (splitTD->tileNumEnd - 1); // 将尾数彻底分给最后一个核心的最后一个tile
            }
        }
    }
}


template <typename T>
void gen_splitTiling(size_t totalLength, uint32_t coreNum, uint8_t* tiling) {
    SplitTilingData splitTD;
    compute_splitTiling_v2<T>(totalLength, coreNum, &splitTD);
    memcpy(tiling, &splitTD, sizeof(SplitTilingData));
}

template <typename T>
void gen_splitTiling(size_t totalLength, uint32_t coreNum, SplitTilingData* splitTD) {
    compute_splitTiling_v2<T>(totalLength, coreNum, splitTD);
}

#endif