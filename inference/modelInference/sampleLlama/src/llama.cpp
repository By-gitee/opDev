#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include "common.h"
#include "struct.h"
#include <sstream>
#include <unordered_map>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#include "aclnnop/aclnn_silu.h"
#include "aclnnop/aclnn_prompt_flash_attention_v2.h"
#include "aclnnop/aclnn_incre_flash_attention_v2.h"
#include "aclnnop/aclnn_slice.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_gather_v2.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_sqrt.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_argmax.h"

bool g_isDevice = false;
using namespace AclnnLlama;
int64_t maxInferNum = 32;
int64_t maxInputLegth = 32;
CommonInfo commonInfo;
aclError ExcuteRope2nd(aclTensor* ropeInput, void** ropeOutputDev, int64_t posIndex, aclrtStream& stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = 4096 * sizeof(uint16_t);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // /* part b */
    // ropeInput - > slice1Output
    int64_t slice1Dim = -1;
    int64_t slice1Start = 0;
    int64_t slice1End = 64;
    int64_t slice1Step = 1;
    std::vector<int64_t> slice1OutputShape = {1, 32, 1, 64};
    aclTensor* slice1Output = nullptr;
    ret = CreateAclTensorWithData(slice1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &slice1Output);
    ret = aclnnSliceGetWorkspaceSize(ropeInput, slice1Dim, slice1Start, slice1End, slice1Step, slice1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // /* part c */
    // ropeInput - > slice2Output
    int64_t slice2Dim = -1;
    int64_t slice2Start = 64;
    int64_t slice2End = 128;
    int64_t slice2Step = 1;
    std::vector<int64_t> slice2OutputShape = {1, 32, 1, 64};
    aclTensor* slice2Output = nullptr;
    ret = CreateAclTensorWithData(slice2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &slice2Output);
    ret = aclnnSliceGetWorkspaceSize(ropeInput, slice2Dim, slice2Start, slice2End, slice2Step, slice2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // slice2Output - > negOutput
    std::vector<int64_t> negOutputShape = {1, 32, 1, 64};
    aclTensor* negOutput = nullptr;
    ret = CreateAclTensorWithData(negOutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &negOutput);
    ret = aclnnNegGetWorkspaceSize(slice2Output, negOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNegGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNeg(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeg failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(slice2Output);

    // negOutput + slice1Output -> catOutput
    int64_t catDim = -1;
    std::vector<int64_t> catOutputShape = {1, 32, 1, 128};
    aclTensor* catOutput = nullptr;
    ret = CreateAclTensorWithData(catOutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &catOutput);
    std::vector<aclTensor*> tmp{negOutput, slice1Output};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    ret = aclnnCatGetWorkspaceSize(tensorList, catDim, catOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(slice1Output);
    aclDestroyTensor(negOutput);

    // /* part a -sin */
    // sinInput + sinIndex - > sinOutput
    std::vector<int64_t> sinIndexShape = {1, 1};
    void* sinIndexDev = nullptr;
    aclTensor* sinIndex = nullptr;
    ret = CreateAclTensor(posIndex, sinIndexShape, &sinIndexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &sinIndex);
    std::vector<int64_t> sinInputShape = {2048, 128};
    aclTensor* sinInput = nullptr;
    ret = CreateAclTensorWithData(sinInputShape, &commonInfo.sinInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &sinInput);
    std::vector<int64_t> sinOutputShape = {1, 1, 128};
    aclTensor* sinOutput = nullptr;
    ret = CreateAclTensorWithData(sinOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &sinOutput);
    int64_t sinDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(sinInput, sinDim, sinIndex, sinOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(sinInput);
    aclDestroyTensor(sinIndex);
    aclrtFree(sinIndexDev);
    sinIndexDev = nullptr;

    // sinOutput - > usq1Output
    std::vector<int64_t> usq1OutputShape = {1, 1, 1, 128};
    std::vector<int64_t> usq1Strides = {128, 128, 128, 1};
    aclTensor* usq1Output = aclCreateTensor(usq1OutputShape.data(), usq1OutputShape.size(), aclDataType::ACL_FLOAT16, usq1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                usq1OutputShape.data(), usq1OutputShape.size(), maxDev_a);

    // catOutput * usq1Output - > mul1Output
    std::vector<int64_t> mul1OutputShape = {1, 32, 1, 128};
    aclTensor* mul1Output = nullptr;   
    ret = CreateAclTensorWithData(mul1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mul1Output);
    ret = aclnnMulGetWorkspaceSize(catOutput, usq1Output, mul1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(catOutput);
    aclDestroyTensor(usq1Output);
    aclDestroyTensor(sinOutput);

    // /* part d - cos */
    // cosIndex + cosInput - > cosOutput
    std::vector<int64_t> cosIndexShape = {1, 1};
    void* cosIndexDev = nullptr;
    aclTensor* cosIndex = nullptr;
    ret = CreateAclTensor(posIndex, cosIndexShape, &cosIndexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &cosIndex);
    std::vector<int64_t> cosInputShape = {2048, 128};
    aclTensor* cosInput = nullptr;
    ret = CreateAclTensorWithData(cosInputShape, &commonInfo.cosInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cosInput);
    std::vector<int64_t> cosOutputShape = {1, 1, 128};
    aclTensor* cosOutput = nullptr;
    ret = CreateAclTensorWithData(cosOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cosOutput);
    int64_t cosDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(cosInput, cosDim, cosIndex, cosOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(cosInput);
    aclDestroyTensor(cosIndex);
    aclrtFree(cosIndexDev);
    cosIndexDev = nullptr;

    // cosOutput - > usq2Output
    std::vector<int64_t> usq2OutputShape = {1, 1, 1, 128};
    std::vector<int64_t> usq2Strides = {128, 128, 128, 1};
    aclTensor* usq2Output = aclCreateTensor(usq2OutputShape.data(), usq2OutputShape.size(), aclDataType::ACL_FLOAT16, usq2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                usq2OutputShape.data(), usq2OutputShape.size(), maxDev_a);

    
    // ropeInput * usq2Output -> mul2Output
    std::vector<int64_t> mul2OutputShape = {1, 32, 1, 128};
    aclTensor* mul2Output = nullptr; 
    ret = CreateAclTensorWithData(mul2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mul2Output);
    ret = aclnnMulGetWorkspaceSize(ropeInput, usq2Output, mul2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(usq2Output);
    aclDestroyTensor(cosOutput);

    // mul1Output + mul2Output - > addOutput
    aclScalar* addAlpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> addOutputShape = {1, 32, 1, 128};
    aclTensor* addOutput = nullptr;
    ret = CreateAclTensorWithData(addOutputShape, ropeOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &addOutput);
    ret = aclnnAddGetWorkspaceSize(mul1Output, mul2Output, addAlpha, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(mul1Output);
    aclDestroyTensor(mul2Output);
    aclDestroyScalar(addAlpha);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    
    return ACL_SUCCESS;
}

aclError ExcuteRmsnorm2nd(aclTensor* rmsnormInput, void** rmsnormOutputDev, void* rmsWeightDev, aclrtStream& stream) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = 4096 * sizeof(float);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    
    // rmsnormInput - > cast1Output
    std::vector<int64_t> cast1OutputShape = {1, 1, 4096};
    aclTensor* cast1Output = nullptr;
    ret = CreateAclTensorWithData(cast1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &cast1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(rmsnormInput, aclDataType::ACL_FLOAT, cast1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // cast1Output - > powOutput
    float powExponentValue = 2.0f;
    std::vector<int64_t> powOutputShape = {1, 1, 4096};
    aclTensor* powOutput = nullptr;
    aclScalar* powExponent = nullptr;
    powExponent = aclCreateScalar(&powExponentValue, aclDataType::ACL_FLOAT);
    ret = CreateAclTensorWithData(powOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &powOutput);
    ret = aclnnPowTensorScalarGetWorkspaceSize(cast1Output, powExponent, powOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyScalar(powExponent);

    // powOutput - > meanOutput
    bool keepdim = true;
    std::vector<int64_t> meanOutputShape = {1, 1, 1};
    aclTensor* meanOutput = nullptr;
    std::vector<int64_t> meanDimData = {-1};
    aclIntArray* meanDim = nullptr;
    meanDim = aclCreateIntArray(meanDimData.data(), meanDimData.size());
    ret = CreateAclTensorWithData(meanOutputShape, &maxDev_c, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &meanOutput);
    ret = aclnnMeanGetWorkspaceSize(powOutput, meanDim, keepdim, aclDataType::ACL_FLOAT, meanOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMean(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMean failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyIntArray(meanDim);
    aclDestroyTensor(powOutput);
    // meanOutput - > addOutput
    float addAlphaValue = 0.000001;
    float addConstValue = 1;
    std::vector<int64_t> addOutputShape = {1, 1, 1};
    aclScalar* addConst = nullptr;
    aclTensor* addOutput = nullptr;
    aclScalar* addAlpha = nullptr;
    addAlpha = aclCreateScalar(&addAlphaValue, aclDataType::ACL_FLOAT);
    addConst = aclCreateScalar(&addConstValue, aclDataType::ACL_FLOAT);
    ret = CreateAclTensorWithData(addOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &addOutput);
    ret = aclnnAddsGetWorkspaceSize(meanOutput, addAlpha, addConst, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdds failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(meanOutput);
    aclDestroyScalar(addConst);
    aclDestroyScalar(addAlpha);

    // addOutput - > addOutput
    ret = aclnnInplaceSqrtGetWorkspaceSize(addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSqrtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceSqrt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSqrt failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // cast1Output / addOutput - > cast1Output
    ret = aclnnInplaceDivGetWorkspaceSize(cast1Output, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceDiv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDiv failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(addOutput);

    // cast1Output - > cast2Output
    std::vector<int64_t> cast2OutputShape = {1, 1, 4096};
    aclTensor* cast2Output = nullptr;
    ret = CreateAclTensorWithData(cast2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cast2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(cast1Output, aclDataType::ACL_FLOAT16, cast2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(cast1Output);

    // cast2Output - > mulOutput
    std::vector<int64_t> mulWeightShape = {1, 1, 4096};
    std::vector<int64_t> mulOutputShape = {1, 1, 4096};
    aclTensor* mulWeight = nullptr;
    aclTensor* mulOutput = nullptr;
    ret = CreateAclTensorWithData(mulWeightShape, &rmsWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulWeight);
    ret = CreateAclTensorWithData(mulOutputShape, rmsnormOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulOutput);
    ret = aclnnMulGetWorkspaceSize(cast2Output, mulWeight, mulOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(mulWeight);
    aclDestroyTensor(cast2Output);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteDecoderLayer2nd(aclTensor* decoderLayerInput, void** decoderLayerOutputDev, DecoderLayerInfo& decoderLayerInfo, int64_t posIndex, aclrtStream& stream) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize_a = 4096 * sizeof(uint16_t);
    auto maxDevSize_b = 4096 * (maxInputLegth + maxInferNum) * sizeof(uint16_t);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    void* maxDev_d = nullptr;
    void* maxDev_e = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize_b, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize_b, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize_b, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_d, maxDevSize_b, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_e, maxDevSize_b, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // decoderLayerInput - >  rmsnorm1Output
    std::vector<int64_t> rmsnorm1OutputShape = {1, 1, 4096};
    aclTensor* rmsnorm1Output = nullptr;
    ret = CreateAclTensorWithData(rmsnorm1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnorm1Output);    
    ret = ExcuteRmsnorm2nd(decoderLayerInput, &maxDev_a, decoderLayerInfo.headRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm2nd failed. ERROR: %d\n", ret); return ret);

    /*first branch*/
    // rmsnorm1Output - > matmul1Output
    std::vector<int64_t> matmul1WeightShape = {4096, 4096};
    std::vector<int64_t> matmul1OutputShape = {1, 1, 4096};
    aclTensor* matmul1Weight = nullptr;
    aclTensor* matmul1Output = nullptr;
    int8_t matmul1CubeMathType=0;
    ret = CreateAclTensorWithData(matmul1WeightShape, &decoderLayerInfo.matmul1WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul1Weight);
    ret = CreateAclTensorWithData(matmul1OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul1Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul1Weight, matmul1Output, matmul1CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul1Weight);

    // matmul1Output(reshape1Input) -> reshape1Output
    std::vector<int64_t> reshape1OutputShape = {1, 1, 32, 128};
    std::vector<int64_t> reshape1Strides = {4096, 4096, 128, 1};
    aclTensor* reshape1Input = aclCreateTensor(reshape1OutputShape.data(), reshape1OutputShape.size(), aclDataType::ACL_FLOAT16, reshape1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape1OutputShape.data(), reshape1OutputShape.size(), maxDev_b);
    aclTensor* reshape1Output = nullptr;
    ret = CreateAclTensorWithData(reshape1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape1Output, reshape1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape1Input);
    aclDestroyTensor(matmul1Output);

    // reshape1Output(trans1Input) -> trans1Output
    std::vector<int64_t> transpose1OutputShape = {1, 32, 1, 128};
    std::vector<int64_t> transpose1Strides = {4096, 128, 4096, 1};
    aclTensor* trans1Input = aclCreateTensor(transpose1OutputShape.data(), transpose1OutputShape.size(), aclDataType::ACL_FLOAT16, transpose1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose1OutputShape.data(), transpose1OutputShape.size(), maxDev_c);

    aclTensor* trans1Output = nullptr;
    ret = CreateAclTensorWithData(transpose1OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans1Output, trans1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans1Input);
    aclDestroyTensor(reshape1Output);
    // rope*1
    // trans1Output -> rope1Output
    std::vector<int64_t> rope1OutputShape = {1, 32, 1, 128};
    aclTensor* rope1Output = nullptr;
    ret = CreateAclTensorWithData(rope1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rope1Output);
    ret = ExcuteRope2nd(trans1Output, &maxDev_c, posIndex, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRope2nd failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(trans1Output);
    /*second branch*/
    // rmsnorm1Output -> matmul2Output
    std::vector<int64_t> matmul2WeightShape = {4096, 4096};
    std::vector<int64_t> matmul2OutputShape = {1, 1, 4096};
    aclTensor* matmul2Weight = nullptr;
    aclTensor* matmul2Output = nullptr;
    int8_t matmul2CubeMathType=0;
    ret = CreateAclTensorWithData(matmul2WeightShape, &decoderLayerInfo.matmul2WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul2Weight);
    ret = CreateAclTensorWithData(matmul2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul2Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul2Weight, matmul2Output, matmul2CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul2Weight);

    // matmul2Output(reshape2Input) -> reshape2Output
    std::vector<int64_t> reshape2OutputShape = {1, 1, 32, 128};
    std::vector<int64_t> reshape2Strides = {4096, 4096, 128, 1};
    aclTensor* reshape2Input = aclCreateTensor(reshape2OutputShape.data(), reshape2OutputShape.size(), aclDataType::ACL_FLOAT16, reshape2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape2OutputShape.data(), reshape2OutputShape.size(), maxDev_b);
    aclTensor* reshape2Output = nullptr;
    ret = CreateAclTensorWithData(reshape2OutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape2Output, reshape2Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape2Input);
    aclDestroyTensor(matmul2Output);

    // reshape2Output(trans2Input) -> trans2Output
    std::vector<int64_t> transpose2OutputShape = {1, 32, 1, 128};
    std::vector<int64_t> transpose2Strides = {4096, 128, 4096, 1};
    aclTensor* trans2Input = aclCreateTensor(transpose2OutputShape.data(), transpose2OutputShape.size(), aclDataType::ACL_FLOAT16, transpose2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose2OutputShape.data(), transpose2OutputShape.size(), maxDev_d);
    aclTensor* trans2Output = nullptr;
    ret = CreateAclTensorWithData(transpose2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans2Output, trans2Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans2Input);
    aclDestroyTensor(reshape2Output);
    // rope*2
    // trans2Output -> rope2Output
    std::vector<int64_t> rope2OutputShape = {1, 32, 1, 128};
    aclTensor* rope2Output = nullptr;
    ret = CreateAclTensorWithData(rope2OutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rope2Output);    
    ret = ExcuteRope2nd(trans2Output, &maxDev_d, posIndex, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRope2nd failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(trans2Output);
    std::vector<int64_t> lastKeyStatesShape = {1, 32, posIndex, 128};
    aclTensor* lastKeyStates = nullptr;
    ret = CreateAclTensorWithData(lastKeyStatesShape, decoderLayerInfo.keyStateMemList, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &lastKeyStates);
    int64_t catKeyDim = 2;
    std::vector<int64_t> catKeyOutputShape = {1, 32, posIndex+1, 128};
    aclTensor* catKeyOutput = nullptr;
    ret = CreateAclTensorWithData(catKeyOutputShape, &maxDev_e, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &catKeyOutput);
    std::vector<aclTensor*> catKeyTmp{lastKeyStates, rope2Output};
    aclTensorList* catKeyTensorList = aclCreateTensorList(catKeyTmp.data(), catKeyTmp.size());
    ret = aclnnCatGetWorkspaceSize(catKeyTensorList, catKeyDim, catKeyOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    // catKeyOutput -> keyStateMemList    
    auto catKeySize = GetShapeSize(catKeyOutputShape) * sizeof(uint16_t);
    ret = aclrtMemcpy(*(decoderLayerInfo.keyStateMemList), catKeySize, maxDev_e, catKeySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to device failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(lastKeyStates);
    aclDestroyTensor(rope2Output);

    /*third branch*/
    // rmsnorm1Output -> matmul3Output
    std::vector<int64_t> matmul3WeightShape = {4096, 4096};
    std::vector<int64_t> matmul3OutputShape = {1, 1, 4096};
    aclTensor* matmul3Weight = nullptr;
    aclTensor* matmul3Output = nullptr;
    int8_t matmul3CubeMathType=0;
    ret = CreateAclTensorWithData(matmul3WeightShape, &decoderLayerInfo.matmul3WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul3Weight);
    ret = CreateAclTensorWithData(matmul3OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul3Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul3Weight, matmul3Output, matmul3CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul3Weight);
    aclDestroyTensor(rmsnorm1Output);

    // matmul3Output(reshape3Input) -> reshape3Output
    std::vector<int64_t> reshape3OutputShape = {1, 1, 32, 128};
    std::vector<int64_t> reshape3Strides = {4096, 4096, 128, 1};
    aclTensor* reshape3Input = aclCreateTensor(reshape3OutputShape.data(), reshape3OutputShape.size(), aclDataType::ACL_FLOAT16, reshape3Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape3OutputShape.data(), reshape3OutputShape.size(), maxDev_b);
    aclTensor* reshape3Output = nullptr;
    ret = CreateAclTensorWithData(reshape3OutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape3Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape3Output, reshape3Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape3Input);
    aclDestroyTensor(matmul3Output);
    // reshape3Output(trans3Input) -> trans3Output
    std::vector<int64_t> transpose3OutputShape = {1, 32, 1, 128};
    std::vector<int64_t> transpose3Strides = {4096, 128, 4096, 1};
    aclTensor* trans3Input = aclCreateTensor(transpose3OutputShape.data(), transpose3OutputShape.size(), aclDataType::ACL_FLOAT16, transpose3Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose3OutputShape.data(), transpose3OutputShape.size(), maxDev_d);
    aclTensor* trans3Output = nullptr;
    ret = CreateAclTensorWithData(transpose3OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans3Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans3Output, trans3Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans3Input);
    aclDestroyTensor(reshape3Output);
   
    // read last v
    // lastValueStates + trans3Output -> catValueOutput
    std::vector<int64_t> lastValueStatesShape = {1, 32, posIndex, 128};
    aclTensor* lastValueStates = nullptr;
    ret = CreateAclTensorWithData(lastValueStatesShape, decoderLayerInfo.valueStateMemList, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &lastValueStates);

    int64_t catValueDim = 2;
    std::vector<int64_t> catValueOutputShape = {1, 32, posIndex + 1, 128};
    aclTensor* catValueOutput = nullptr;
    ret = CreateAclTensorWithData(catValueOutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &catValueOutput);
    std::vector<aclTensor*> catValueTmp{lastValueStates, trans3Output};
    aclTensorList* catValueTensorList = aclCreateTensorList(catValueTmp.data(), catValueTmp.size());
    ret = aclnnCatGetWorkspaceSize(catValueTensorList, catValueDim, catValueOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    // catValueOutput -> keyStateMemList    
    auto catValueSize = GetShapeSize(catValueOutputShape) * sizeof(uint16_t);
    ret = aclrtMemcpy(*(decoderLayerInfo.valueStateMemList), catValueSize, maxDev_d, catValueSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to device failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(lastValueStates);
    aclDestroyTensor(trans3Output);

    // rope1Output + catKeyOutput + catValueOutput -> attentionOut
    std::vector<aclTensor*> keyTmp{catKeyOutput};
    aclTensorList* keyList = aclCreateTensorList(keyTmp.data(), keyTmp.size());
    std::vector<aclTensor*> valueTmp{catValueOutput};
    aclTensorList* valueList = aclCreateTensorList(valueTmp.data(), valueTmp.size());
    aclTensor* paddingMask = nullptr;
    aclTensor* attenMask = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
    aclTensor* dequantScale1 = nullptr;
    aclTensor* quantScale1 = nullptr;
    aclTensor* dequantScale2 = nullptr;
    aclTensor* quantScale2 = nullptr;
    aclTensor* quantOffset2 = nullptr;
    int64_t numHeads = 32;
    double scaleValue = 1.0 / sqrt (128.0);
    char inputLayout[] = "BNSD";
    int64_t numKeyValueHeads = 32;
    std::vector<int64_t> attentionOutShape = {1, 32, 1, 128};
    aclTensor* attentionOut = nullptr;
    ret = CreateAclTensorWithData(attentionOutShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &attentionOut);
    ret = aclnnIncreFlashAttentionV2GetWorkspaceSize(rope1Output, keyList, valueList,
            paddingMask, attenMask, actualSeqLengths, dequantScale1, quantScale1, dequantScale2, quantScale2, quantOffset2,
            numHeads, scaleValue, inputLayout, numKeyValueHeads,
            attentionOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnIncreFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIncreFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(rope1Output);
    aclDestroyTensor(catKeyOutput);
    aclDestroyTensor(catValueOutput);

    // attentionOut(trans4Input) -> trans4Output
    std::vector<int64_t> transpose4OutputShape = {1, 1, 32, 128};
    std::vector<int64_t> transpose4Strides = {4096, 128, 128, 1};
    aclTensor* trans4Input = aclCreateTensor(transpose4OutputShape.data(), transpose4OutputShape.size(), aclDataType::ACL_FLOAT16, transpose4Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose4OutputShape.data(), transpose4OutputShape.size(), maxDev_a);
    aclTensor* trans4Output = nullptr;
    ret = CreateAclTensorWithData(transpose4OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans4Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans4Output, trans4Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans4Input);
    aclDestroyTensor(attentionOut);

    // trans4Output(reshape4Input) -> reshape4Output
    std::vector<int64_t> reshape4OutputShape = {1, 1, 4096};
    std::vector<int64_t> reshape4Strides = {4096, 4096, 1};
    aclTensor* reshape4Input = aclCreateTensor(reshape4OutputShape.data(), reshape4OutputShape.size(), aclDataType::ACL_FLOAT16, reshape4Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape4OutputShape.data(), reshape4OutputShape.size(), maxDev_b);
    aclTensor* reshape4Output = nullptr;
    ret = CreateAclTensorWithData(reshape4OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape4Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape4Output, reshape4Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape4Input);
    aclDestroyTensor(trans4Output);

    // reshape4Output -> matmul4Output
    std::vector<int64_t> matmul4WeightShape = {4096, 4096};
    std::vector<int64_t> matmul4OutputShape = {1, 1, 4096};
    aclTensor* matmul4Weight = nullptr;
    aclTensor* matmul4Output = nullptr;
    int8_t matmul4CubeMathType=0;
    ret = CreateAclTensorWithData(matmul4WeightShape, &decoderLayerInfo.matmul4WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul4Weight);
    ret = CreateAclTensorWithData(matmul4OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul4Output);
    ret = aclnnMatmulGetWorkspaceSize(reshape4Output, matmul4Weight, matmul4Output, matmul4CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul4Weight);
    aclDestroyTensor(reshape4Output);

    // decoderLayerInput + matmul4Output -> add1Output
    aclScalar* add1Alpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> add1OutputShape = {1, 1, 4096};
    aclTensor* add1Output = nullptr;
    ret = CreateAclTensorWithData(add1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &add1Output);
    ret = aclnnAddGetWorkspaceSize(decoderLayerInput, matmul4Output, add1Alpha, add1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyScalar(add1Alpha);
    aclDestroyTensor(matmul4Output);
    // rmsnorm*2
    // add1Output -> rmsnorm2Output
    std::vector<int64_t> rmsnorm2OutputShape = {1, 1, 4096};
    // void* rmsnorm2Output
    aclTensor* rmsnorm2Output = nullptr;
    ret = CreateAclTensorWithData(rmsnorm2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnorm2Output);    
    ret = ExcuteRmsnorm2nd(add1Output, &maxDev_b, decoderLayerInfo.tailRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm2nd failed. ERROR: %d\n", ret); return ret);
    
    // rmsnorm2Output -> matmul5Output
    std::vector<int64_t> matmul5WeightShape = {4096, 11008};
    std::vector<int64_t> matmul5OutputShape = {1, 1, 11008};
    aclTensor* matmul5Weight = nullptr;
    aclTensor* matmul5Output = nullptr;    
    int8_t matmul5CubeMathType=0;
    ret = CreateAclTensorWithData(matmul5WeightShape, &decoderLayerInfo.matmul5WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul5Weight);
    ret = CreateAclTensorWithData(matmul5OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul5Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm2Output, matmul5Weight, matmul5Output, matmul5CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul5Weight);

    // matmul5Output -> siluOutput
    std::vector<int64_t> siluOutputShape = {1, 1, 11008};
    aclTensor* siluOutput = nullptr;
    ret = CreateAclTensorWithData(siluOutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &siluOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclnnSiluGetWorkspaceSize(matmul5Output, siluOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilu failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul5Output);

    // rmsnorm2Output -> matmul6Output
    std::vector<int64_t> matmul6WeightShape = {4096,11008};
    std::vector<int64_t> matmul6OutputShape = {1, 1, 11008};
    aclTensor* matmul6Weight = nullptr;
    aclTensor* matmul6Output = nullptr;
    int8_t matmul6CubeMathType=0;
    ret = CreateAclTensorWithData(matmul6WeightShape, &decoderLayerInfo.matmul6WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul6Weight);
    ret = CreateAclTensorWithData(matmul6OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul6Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm2Output, matmul6Weight, matmul6Output, matmul6CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul6Weight);
    aclDestroyTensor(rmsnorm2Output);

    // matmul6Output * siluOutput -> mulOutput
    std::vector<int64_t> mulOutputShape = {1, 1, 11008};
    aclTensor* mulOutput = nullptr;
    ret = CreateAclTensorWithData(mulOutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulOutput);
    ret = aclnnMulGetWorkspaceSize(matmul6Output, siluOutput, mulOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul6Output);
    aclDestroyTensor(siluOutput);

    // mulOutput -> matmul7Output
    std::vector<int64_t> matmul7WeightShape = {11008, 4096};
    std::vector<int64_t> matmul7OutputShape = {1, 1, 4096};
    aclTensor* matmul7Weight = nullptr;
    aclTensor* matmul7Output = nullptr;
    int8_t matmul7CubeMathType=0;
    ret = CreateAclTensorWithData(matmul7WeightShape, &decoderLayerInfo.matmul7WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul7Weight);
    ret = CreateAclTensorWithData(matmul7OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul7Output);
    ret = aclnnMatmulGetWorkspaceSize(mulOutput, matmul7Weight, matmul7Output, matmul7CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul7Weight);
    aclDestroyTensor(mulOutput);

    // matmul7Output + add1Output-> add2Output
    aclScalar* add2Alpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> add2OutputShape = {1, 1, 4096};
    aclTensor* add2Output = nullptr;
    ret = CreateAclTensorWithData(add2OutputShape, decoderLayerOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &add2Output);
    ret = aclnnAddGetWorkspaceSize(matmul7Output, add1Output, add2Alpha, add2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul7Output);
    aclDestroyTensor(add1Output);
    aclDestroyScalar(add2Alpha);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    aclrtFree(maxDev_d);
    maxDev_d = nullptr;
    aclrtFree(maxDev_e);
    maxDev_e = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteLlama2nd(aclTensor* llama2ndInput, void** llama2ndOutputDev, LlamaInfo& llamaInfo, aclrtStream& stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    auto maxDevSize = 1 * 32001 * sizeof(float);
    void* maxDev_a = nullptr; 
    void* maxDev_b = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> gatherWeightShape = {32001, 4096};
    std::vector<int64_t> gatherOutputShape = {1, 1, 4096};
    aclTensor* gatherWeight = nullptr;
    aclTensor* gatherOutput = nullptr;
    ret = CreateAclTensorWithData(gatherWeightShape, &llamaInfo.embedTokensWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &gatherWeight);
    ret = CreateAclTensorWithData(gatherOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &gatherOutput);
    int64_t sinDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(gatherWeight, sinDim, llama2ndInput, gatherOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(gatherWeight);
    
    for(int i=0; i < 32; i++) {
        ret = ExcuteDecoderLayer2nd(gatherOutput, &maxDev_a, llamaInfo.decoderLayerInfo[i], llamaInfo.posIndex, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteDecoderLayer2nd failed. ERROR: %d\n", ret); return ret);
    }

    std::vector<int64_t> rmsnormOutputShape = {1, 1, 4096};
    aclTensor* rmsnormOutput = nullptr;
    ret = CreateAclTensorWithData(rmsnormOutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnormOutput);    
    ret = ExcuteRmsnorm2nd(gatherOutput, &maxDev_b, llamaInfo.lastRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm2nd failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(gatherOutput);

    std::vector<int64_t> matmulWeightShape = {4096, 32001};
    std::vector<int64_t> matmulOutputShape = {1, 1, 32001};
    aclTensor* matmulWeight = nullptr;
    aclTensor* matmulOutput = nullptr;
    int8_t matmulCubeMathType=0;
    ret = CreateAclTensorWithData(matmulWeightShape, &llamaInfo.lmHeadWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmulWeight);
    ret = CreateAclTensorWithData(matmulOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmulOutput);
    ret = aclnnMatmulGetWorkspaceSize(rmsnormOutput, matmulWeight, matmulOutput, matmulCubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(rmsnormOutput);
    aclDestroyTensor(matmulWeight);

    std::vector<int64_t> castOutputShape = {1, 1, 32001};
    aclTensor* castOutput = nullptr;
    ret = CreateAclTensorWithData(castOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &castOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(matmulOutput, aclDataType::ACL_FLOAT, castOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmulOutput);

    std::vector<int64_t> argMaxOutputShape = {1, 1, 1};
    void* argMaxOutputDev = nullptr;
    aclTensor* argMaxOutput = nullptr;
    ret = CreateAclTensor(argMaxOutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &argMaxOutput);
    int64_t argMaxDim = -1;
    bool argMaxKeepdim = true;
    ret = aclnnArgMaxGetWorkspaceSize(castOutput, argMaxDim, argMaxKeepdim, argMaxOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnArgMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgMax failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(castOutput);

    std::vector<int64_t> reshape1OutputShape = {1, 1};
    aclTensor* reshape1Input = nullptr;
    aclTensor* reshape1Output = nullptr;
    ret = CreateAclTensorWithData(reshape1OutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &reshape1Input);
    ret = CreateAclTensorWithData(reshape1OutputShape, llama2ndOutputDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &reshape1Output);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape1Output, reshape1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape1Input);
    aclDestroyTensor(argMaxOutput);
    aclrtFree(argMaxOutputDev);
    argMaxOutputDev = nullptr;
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteRope(aclTensor* ropeInput, void** ropeOutputDev, aclrtStream& stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = maxInputLegth * 32 * 128 * sizeof(uint16_t);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // /* part b */
    // ropeInput - > slice1Output
    int64_t slice1Dim = -1;
    int64_t slice1Start = 0;
    int64_t slice1End = 64;
    int64_t slice1Step = 1;
    std::vector<int64_t> slice1OutputShape = {1, 32, maxInputLegth, 64};
    aclTensor* slice1Output = nullptr;
    ret = CreateAclTensorWithData(slice1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &slice1Output);
    ret = aclnnSliceGetWorkspaceSize(ropeInput, slice1Dim, slice1Start, slice1End, slice1Step, slice1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // /* part c */
    // ropeInput - > slice2Output
    int64_t slice2Dim = -1;
    int64_t slice2Start = 64;
    int64_t slice2End = 128;
    int64_t slice2Step = 1;
    std::vector<int64_t> slice2OutputShape = {1, 32, maxInputLegth, 64};
    aclTensor* slice2Output = nullptr;
    ret = CreateAclTensorWithData(slice2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &slice2Output);
    ret = aclnnSliceGetWorkspaceSize(ropeInput, slice2Dim, slice2Start, slice2End, slice2Step, slice2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSlice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // slice2Output - > negOutput
    std::vector<int64_t> negOutputShape = {1, 32, maxInputLegth, 64};
    aclTensor* negOutput = nullptr;
    ret = CreateAclTensorWithData(negOutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &negOutput);
    ret = aclnnNegGetWorkspaceSize(slice2Output, negOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNegGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNeg(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNeg failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(slice2Output);

    // negOutput + slice1Output -> catOutput
    int64_t catDim = -1;
    std::vector<int64_t> catOutputShape = {1, 32, maxInputLegth, 128};
    aclTensor* catOutput = nullptr;
    ret = CreateAclTensorWithData(catOutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &catOutput);
    std::vector<aclTensor*> tmp{negOutput, slice1Output};
    aclTensorList* tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    ret = aclnnCatGetWorkspaceSize(tensorList, catDim, catOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(slice1Output);
    aclDestroyTensor(negOutput);

    // /* part a -sin */
    // sinInput + sinIndex - > sinOutput
    std::vector<int64_t> sinInputShape = {2048, 128};
    std::vector<int64_t> sinIndexShape = {1, maxInputLegth};
    std::vector<int64_t> sinOutputShape = {1, maxInputLegth, 128};
    aclTensor* sinInput = nullptr;
    aclTensor* sinIndex = nullptr;
    aclTensor* sinOutput = nullptr;
    ret = CreateAclTensorWithData(sinInputShape, &commonInfo.sinInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &sinInput);
    ret = CreateAclTensorWithData(sinIndexShape, &commonInfo.indexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &sinIndex);
    ret = CreateAclTensorWithData(sinOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &sinOutput);
    int64_t sinDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(sinInput, sinDim, sinIndex, sinOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(sinInput);
    aclDestroyTensor(sinIndex);

    // sinOutput - > usq1Output
    std::vector<int64_t> usq1OutputShape = {1, 1, maxInputLegth, 128};
    std::vector<int64_t> usq1Strides = {maxInputLegth*128, maxInputLegth*128, 128, 1};
    aclTensor* usq1Output = aclCreateTensor(usq1OutputShape.data(), usq1OutputShape.size(), aclDataType::ACL_FLOAT16, usq1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                usq1OutputShape.data(), usq1OutputShape.size(), maxDev_a);

    // catOutput * usq1Output - > mul1Output
    std::vector<int64_t> mul1OutputShape = {1, 32, maxInputLegth, 128};
    aclTensor* mul1Output = nullptr;   
    ret = CreateAclTensorWithData(mul1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mul1Output);
    ret = aclnnMulGetWorkspaceSize(catOutput, usq1Output, mul1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(catOutput);
    aclDestroyTensor(usq1Output);
    aclDestroyTensor(sinOutput);

    // /* part d - cos */
    // cosIndex + cosInput - > cosOutput
    std::vector<int64_t> cosInputShape = {2048, 128};
    std::vector<int64_t> cosIndexShape = {1, maxInputLegth};
    std::vector<int64_t> cosOutputShape = {1, maxInputLegth, 128};
    aclTensor* cosInput = nullptr;
    aclTensor* cosIndex = nullptr;
    aclTensor* cosOutput = nullptr;
    
    ret = CreateAclTensorWithData(cosInputShape, &commonInfo.cosInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cosInput);
    ret = CreateAclTensorWithData(cosIndexShape, &commonInfo.indexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &cosIndex);
    ret = CreateAclTensorWithData(cosOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cosOutput);
    int64_t cosDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(cosInput, cosDim, cosIndex, cosOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(cosInput);
    aclDestroyTensor(cosIndex);

    // cosOutput - > usq2Output
    std::vector<int64_t> usq2OutputShape = {1, 1, maxInputLegth, 128};
    std::vector<int64_t> usq2Strides = {maxInputLegth*128, maxInputLegth*128, 128, 1};
    aclTensor* usq2Output = aclCreateTensor(usq2OutputShape.data(), usq2OutputShape.size(), aclDataType::ACL_FLOAT16, usq2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                usq2OutputShape.data(), usq2OutputShape.size(), maxDev_a);

    // ropeInput * usq2Output -> mul2Output
    std::vector<int64_t> mul2OutputShape = {1, 32, maxInputLegth, 128};
    aclTensor* mul2Output = nullptr; 
    ret = CreateAclTensorWithData(mul2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mul2Output);
    ret = aclnnMulGetWorkspaceSize(ropeInput, usq2Output, mul2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(usq2Output);
    aclDestroyTensor(cosOutput);

    // mul1Output + mul2Output - > addOutput
    aclScalar* addAlpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> addOutputShape = {1, 32, maxInputLegth, 128};
    aclTensor* addOutput = nullptr;
    ret = CreateAclTensorWithData(addOutputShape, ropeOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &addOutput);
    ret = aclnnAddGetWorkspaceSize(mul1Output, mul2Output, addAlpha, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(mul1Output);
    aclDestroyTensor(mul2Output);
    aclDestroyScalar(addAlpha);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteRmsnorm(aclTensor* rmsnormInput, void** rmsnormOutputDev, void* rmsWeightDev, aclrtStream& stream) {

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = maxInputLegth * 32 * 128 * sizeof(float);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // rmsnormInput - > cast1Output
    std::vector<int64_t> cast1OutputShape = {1, maxInputLegth, 4096};
    aclTensor* cast1Output = nullptr;
    ret = CreateAclTensorWithData(cast1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &cast1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(rmsnormInput, aclDataType::ACL_FLOAT, cast1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    // cast1Output - > powOutput
    float powExponentValue = 2.0f;
    std::vector<int64_t> powOutputShape = {1, maxInputLegth, 4096};
    aclTensor* powOutput = nullptr;
    aclScalar* powExponent = nullptr;
    powExponent = aclCreateScalar(&powExponentValue, aclDataType::ACL_FLOAT);
    ret = CreateAclTensorWithData(powOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &powOutput);
    ret = aclnnPowTensorScalarGetWorkspaceSize(cast1Output, powExponent, powOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalarGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnPowTensorScalar(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPowTensorScalar failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyScalar(powExponent);

    // powOutput - > meanOutput
    bool keepdim = true;
    std::vector<int64_t> meanOutputShape = {1, maxInputLegth, 1};
    aclTensor* meanOutput = nullptr;
    std::vector<int64_t> meanDimData = {-1};
    aclIntArray* meanDim = nullptr;
    meanDim = aclCreateIntArray(meanDimData.data(), meanDimData.size());
    ret = CreateAclTensorWithData(meanOutputShape, &maxDev_c, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &meanOutput);
    ret = aclnnMeanGetWorkspaceSize(powOutput, meanDim, keepdim, aclDataType::ACL_FLOAT, meanOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMeanGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMean(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMean failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyIntArray(meanDim);
    aclDestroyTensor(powOutput);
    // meanOutput - > addOutput
    float addAlphaValue = 0.000001;
    float addConstValue = 1;
    std::vector<int64_t> addOutputShape = {1, maxInputLegth, 1};
    aclScalar* addConst = nullptr;
    aclTensor* addOutput = nullptr;
    aclScalar* addAlpha = nullptr;
    addAlpha = aclCreateScalar(&addAlphaValue, aclDataType::ACL_FLOAT);
    addConst = aclCreateScalar(&addConstValue, aclDataType::ACL_FLOAT);
    ret = CreateAclTensorWithData(addOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &addOutput);
    ret = aclnnAddsGetWorkspaceSize(meanOutput, addAlpha, addConst, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddsGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdds(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdds failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(meanOutput);
    aclDestroyScalar(addConst);
    aclDestroyScalar(addAlpha);

    // addOutput - > addOutput
    ret = aclnnInplaceSqrtGetWorkspaceSize(addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceSqrtGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceSqrt(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSqrt failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }

    // cast1Output / addOutput - > cast1Output
    ret = aclnnInplaceDivGetWorkspaceSize(cast1Output, addOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDivGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceDiv(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceDiv failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(addOutput);
    // cast1Output - > cast2Output
    std::vector<int64_t> cast2OutputShape = {1, maxInputLegth, 4096};
    aclTensor* cast2Output = nullptr;
    ret = CreateAclTensorWithData(cast2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &cast2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(cast1Output, aclDataType::ACL_FLOAT16, cast2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(cast1Output);

    // cast2Output - > mulOutput
    std::vector<int64_t> mulWeightShape = {1, 1, 4096};
    std::vector<int64_t> mulOutputShape = {1, maxInputLegth, 4096};
    aclTensor* mulWeight = nullptr;
    aclTensor* mulOutput = nullptr;
    ret = CreateAclTensorWithData(mulWeightShape, &rmsWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulWeight);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithData failed. ERROR: %d\n", ret); return ret);
    ret = CreateAclTensorWithData(mulOutputShape, rmsnormOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulOutput);
    ret = aclnnMulGetWorkspaceSize(cast2Output, mulWeight, mulOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(mulWeight);
    aclDestroyTensor(cast2Output);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteDecoderLayer(aclTensor* decoderLayerInput, void** decoderLayerOutputDev, DecoderLayerInfo& decoderLayerInfo, aclrtStream& stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = maxInputLegth * 11008 * sizeof(uint16_t);
    void* maxDev_a = nullptr;
    void* maxDev_b = nullptr;
    void* maxDev_c = nullptr;
    void* maxDev_d = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_c, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_d, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // decoderLayerInput - >  rmsnorm1Output
    std::vector<int64_t> rmsnorm1OutputShape = {1, maxInputLegth, 4096};
    aclTensor* rmsnorm1Output = nullptr;
    ret = CreateAclTensorWithData(rmsnorm1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnorm1Output);
    ret = ExcuteRmsnorm(decoderLayerInput, &maxDev_a, decoderLayerInfo.headRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm failed. ERROR: %d\n", ret); return ret);

    /*first branch*/
    // rmsnorm1Output - > matmul1Output
    std::vector<int64_t> matmul1WeightShape = {4096, 4096};
    std::vector<int64_t> matmul1OutputShape = {1, maxInputLegth, 4096};
    aclTensor* matmul1Weight = nullptr;
    aclTensor* matmul1Output = nullptr;
    int8_t matmul1CubeMathType=0;
    ret = CreateAclTensorWithData(matmul1WeightShape, &decoderLayerInfo.matmul1WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul1Weight);
    ret = CreateAclTensorWithData(matmul1OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul1Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul1Weight, matmul1Output, matmul1CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul1Weight);

    // matmul1Output(reshape1Input) -> reshape1Output
    std::vector<int64_t> reshape1OutputShape = {1, maxInputLegth, 32, 128};
    std::vector<int64_t> reshape1Strides = {maxInputLegth*4096, 4096, 128, 1};  
    aclTensor* reshape1Input = aclCreateTensor(reshape1OutputShape.data(), reshape1OutputShape.size(), aclDataType::ACL_FLOAT16, reshape1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape1OutputShape.data(), reshape1OutputShape.size(), maxDev_b);
    void* reshape1OutputDev = nullptr;
    aclTensor* reshape1Output = nullptr;
    ret = CreateAclTensorWithData(reshape1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape1Output, reshape1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape1Input);
    aclDestroyTensor(matmul1Output);

    // reshape1Output(trans1Input) -> trans1Output
    std::vector<int64_t> transpose1OutputShape = {1, 32, maxInputLegth, 128};
    std::vector<int64_t> transpose1Strides = {maxInputLegth*4096, 128, 4096, 1};
    aclTensor* trans1Input = aclCreateTensor(transpose1OutputShape.data(), transpose1OutputShape.size(), aclDataType::ACL_FLOAT16, transpose1Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose1OutputShape.data(), transpose1OutputShape.size(), maxDev_c);
    aclTensor* trans1Output = nullptr;
    ret = CreateAclTensorWithData(transpose1OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans1Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans1Output, trans1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans1Input);
    aclDestroyTensor(reshape1Output);

    // trans1Output -> rope1Output
    std::vector<int64_t> rope1OutputShape = {1, 32, maxInputLegth, 128};
    aclTensor* rope1Output = nullptr;
    ret = CreateAclTensorWithData(rope1OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rope1Output);    
    ret = ExcuteRope(trans1Output, &maxDev_c, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRope failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(trans1Output);

    /*second branch*/
    // rmsnorm1Output -> matmul2Output
    std::vector<int64_t> matmul2WeightShape = {4096, 4096};
    std::vector<int64_t> matmul2OutputShape = {1, maxInputLegth, 4096};
    void* matmul2OutputDev = nullptr;
    aclTensor* matmul2Weight = nullptr;
    aclTensor* matmul2Output = nullptr;
    int8_t matmul2CubeMathType=0;
    ret = CreateAclTensorWithData(matmul2WeightShape, &decoderLayerInfo.matmul2WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul2Weight);
    ret = CreateAclTensorWithData(matmul2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul2Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul2Weight, matmul2Output, matmul2CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul2Weight);

    // matmul2Output(reshape2Input) -> reshape2Output
    std::vector<int64_t> reshape2OutputShape = {1, maxInputLegth, 32, 128};
    std::vector<int64_t> reshape2Strides = {maxInputLegth*4096, 4096, 128, 1};
    aclTensor* reshape2Input = aclCreateTensor(reshape2OutputShape.data(), reshape2OutputShape.size(), aclDataType::ACL_FLOAT16, reshape2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape2OutputShape.data(), reshape2OutputShape.size(), maxDev_b);
    void* reshape2OutputDev = nullptr;
    aclTensor* reshape2Output = nullptr;
    ret = CreateAclTensorWithData(reshape2OutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape2Output, reshape2Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape2Input);
    aclDestroyTensor(matmul2Output);

    // reshape2Output(trans2Input) -> trans2Output
    std::vector<int64_t> transpose2OutputShape = {1, 32, maxInputLegth, 128};
    std::vector<int64_t> transpose2Strides = {maxInputLegth*4096, 128, 4096, 1};
    aclTensor* trans2Input = aclCreateTensor(transpose2OutputShape.data(), transpose2OutputShape.size(), aclDataType::ACL_FLOAT16, transpose2Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose2OutputShape.data(), transpose2OutputShape.size(), maxDev_d);
    aclTensor* trans2Output = nullptr;
    ret = CreateAclTensorWithData(transpose2OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans2Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans2Output, trans2Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans2Input);
    aclDestroyTensor(reshape2Output);
    
    // trans2Output -> rope2Output
    std::vector<int64_t> rope2OutputShape = {1, 32, maxInputLegth, 128};
    std::vector<int64_t> rope2OutputStrides = {maxInputLegth*4096, 128*maxInputLegth, 128, 1};
    aclTensor* rope2Output = aclCreateTensor(rope2OutputShape.data(), rope2OutputShape.size(), aclDataType::ACL_FLOAT16, rope2OutputStrides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                rope2OutputShape.data(), rope2OutputShape.size(), *(decoderLayerInfo.keyStateMemList));
    ret = ExcuteRope(trans2Output, decoderLayerInfo.keyStateMemList, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRope failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(trans2Output);

    /*third branch*/
    // rmsnorm1Output -> matmul3Output
    std::vector<int64_t> matmul3WeightShape = {4096, 4096};
    std::vector<int64_t> matmul3OutputShape = {1, maxInputLegth, 4096};
    aclTensor* matmul3Weight = nullptr;
    aclTensor* matmul3Output = nullptr;
    int8_t matmul3CubeMathType=0;
    ret = CreateAclTensorWithData(matmul3WeightShape, &decoderLayerInfo.matmul3WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul3Weight);
    ret = CreateAclTensorWithData(matmul3OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul3Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm1Output, matmul3Weight, matmul3Output, matmul3CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul3Weight);
    aclDestroyTensor(rmsnorm1Output);

    // matmul3Output(reshape3Input) -> reshape3Output
    std::vector<int64_t> reshape3OutputShape = {1, maxInputLegth, 32, 128};
    std::vector<int64_t> reshape3Strides = {maxInputLegth*4096, 4096, 128, 1};
    aclTensor* reshape3Input = aclCreateTensor(reshape3OutputShape.data(), reshape3OutputShape.size(), aclDataType::ACL_FLOAT16, reshape3Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape3OutputShape.data(), reshape3OutputShape.size(), maxDev_b);
    aclTensor* reshape3Output = nullptr;
    ret = CreateAclTensorWithData(reshape3OutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape3Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape3Output, reshape3Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape3Input);
    aclDestroyTensor(matmul3Output);

    // reshape3Output(trans3Input) -> trans3Output
    std::vector<int64_t> transpose3OutputShape = {1, 32, maxInputLegth, 128};
    std::vector<int64_t> transpose3Strides = {maxInputLegth*4096, 128, 4096, 1};
    aclTensor* trans3Input = aclCreateTensor(transpose3OutputShape.data(), transpose3OutputShape.size(), aclDataType::ACL_FLOAT16, transpose3Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose3OutputShape.data(), transpose3OutputShape.size(), maxDev_d);
    aclTensor* trans3Output = nullptr;
    ret = CreateAclTensorWithData(transpose3OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans3Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclnnInplaceCopyGetWorkspaceSize(trans3Output, trans3Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans3Input);
    aclDestroyTensor(reshape3Output);

    auto trans3ValueSize = GetShapeSize(transpose3OutputShape) * sizeof(uint16_t);
    ret = aclrtMemcpy(*(decoderLayerInfo.valueStateMemList), trans3ValueSize, maxDev_b, trans3ValueSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to device failed. ERROR: %d\n", ret); return ret);

    /*fourth branch*/
    std::vector<int64_t> attenMaskShape = {2048, 2048};
    aclTensor* attenMask = nullptr;
    ret = CreateAclTensorWithData(attenMaskShape, &commonInfo.attenMaskDev, aclDataType::ACL_UINT8, aclFormat::ACL_FORMAT_ND, &attenMask);

    // rope1Output + rope2Output + trans3Output + attenMask = attentionOut
    aclTensor* paddingMask = nullptr;
    aclIntArray* actualSeqLengths = nullptr;
    aclIntArray* actualSeqLengthsKv = nullptr;
    aclTensor* deqScale1 = nullptr;
    aclTensor* quantScale1 = nullptr;
    aclTensor* deqScale2 = nullptr;
    aclTensor* quantScale2 = nullptr;
    aclTensor* quantOffset2 = nullptr;
    int64_t numHeads = 32;
    double scaleValue = 1.0 / sqrt (128.0);
    int64_t preTokens = 214748647;
    int64_t nextTokens = 0;
    char inputLayout[] = "BNSD";
    int64_t numKeyValueHeads = 32;
    int64_t sparseMode = 2;
    std::vector<int64_t> attentionOutShape = {1, 32, maxInputLegth, 128};
    aclTensor* attentionOut = nullptr;
    ret = CreateAclTensorWithData(attentionOutShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &attentionOut);
    ret = aclnnPromptFlashAttentionV2GetWorkspaceSize(rope1Output, rope2Output, trans3Output,
                                                      paddingMask, attenMask, actualSeqLengths, actualSeqLengthsKv,
                                                      deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                                                      numHeads, scaleValue, preTokens, nextTokens, inputLayout,
                                                      numKeyValueHeads, sparseMode, attentionOut,
                                                      &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnPromptFlashAttentionV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPromptFlashAttentionV2 failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(rope1Output);
    aclDestroyTensor(rope2Output);
    aclDestroyTensor(trans3Output);
    aclDestroyTensor(attenMask);
    
    // attentionOut(trans4Input) -> trans4Output
    std::vector<int64_t> transpose4OutputShape = {1, maxInputLegth, 32, 128};
    std::vector<int64_t> transpose4Strides = {maxInputLegth*4096, 128, 128*maxInputLegth, 1};
    aclTensor* trans4Input = aclCreateTensor(transpose4OutputShape.data(), transpose4OutputShape.size(), aclDataType::ACL_FLOAT16, transpose4Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                transpose4OutputShape.data(), transpose4OutputShape.size(), maxDev_d);
    aclTensor* trans4Output = nullptr;
    ret = CreateAclTensorWithData(transpose4OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &trans4Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(trans4Output, trans4Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(trans4Input);
    aclDestroyTensor(attentionOut);

    // trans4Output(reshape4Input) -> reshape4Output
    std::vector<int64_t> reshape4OutputShape = {1, maxInputLegth, 4096};
    std::vector<int64_t> reshape4Strides = {maxInputLegth*4096, 4096, 1};
    aclTensor* reshape4Input = aclCreateTensor(reshape4OutputShape.data(), reshape4OutputShape.size(), aclDataType::ACL_FLOAT16, reshape4Strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                reshape4OutputShape.data(), reshape4OutputShape.size(), maxDev_a);
    aclTensor* reshape4Output = nullptr;
    ret = CreateAclTensorWithData(reshape4OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &reshape4Output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape4Output, reshape4Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape4Input);
    aclDestroyTensor(trans4Output);

    // reshape4Output -> matmul4Output    
    std::vector<int64_t> matmul4WeightShape = {4096, 4096};
    std::vector<int64_t> matmul4OutputShape = {1, maxInputLegth, 4096};
    aclTensor* matmul4Weight = nullptr;
    aclTensor* matmul4Output = nullptr;
    int8_t matmul4CubeMathType=0;
    ret = CreateAclTensorWithData(matmul4WeightShape, &decoderLayerInfo.matmul4WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul4Weight);
    ret = CreateAclTensorWithData(matmul4OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul4Output);
    ret = aclnnMatmulGetWorkspaceSize(reshape4Output, matmul4Weight, matmul4Output, matmul4CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul4Weight);
    aclDestroyTensor(reshape4Output);

    // decoderLayerInput + matmul4Output -> add1Output
    aclScalar* add1Alpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> add1OutputShape = {1, maxInputLegth, 4096};
    aclTensor* add1Output = nullptr;
    ret = CreateAclTensorWithData(add1OutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &add1Output);
    ret = aclnnAddGetWorkspaceSize(decoderLayerInput, matmul4Output, add1Alpha, add1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyScalar(add1Alpha);
    aclDestroyTensor(matmul4Output);

    // add1Output -> rmsnorm2Output
    std::vector<int64_t> rmsnorm2OutputShape = {1, maxInputLegth, 4096};
    aclTensor* rmsnorm2Output = nullptr;
    ret = CreateAclTensorWithData(rmsnorm2OutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnorm2Output); 
    ret = ExcuteRmsnorm(add1Output, &maxDev_a, decoderLayerInfo.tailRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm failed. ERROR: %d\n", ret); return ret);

    // rmsnorm2Output -> matmul5Output
    std::vector<int64_t> matmul5WeightShape = {4096, 11008};
    std::vector<int64_t> matmul5OutputShape = {1, maxInputLegth, 11008};
    aclTensor* matmul5Weight = nullptr;
    aclTensor* matmul5Output = nullptr;    
    int8_t matmul5CubeMathType=0;
    ret = CreateAclTensorWithData(matmul5WeightShape, &decoderLayerInfo.matmul5WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul5Weight);
    ret = CreateAclTensorWithData(matmul5OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul5Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm2Output, matmul5Weight, matmul5Output, matmul5CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul5Weight);
    // matmul5Output -> siluOutput
    std::vector<int64_t> siluOutputShape = {1, maxInputLegth, 11008};
    aclTensor* siluOutput = nullptr;
    ret = CreateAclTensorWithData(siluOutputShape, &maxDev_d, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &siluOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclnnSiluGetWorkspaceSize(matmul5Output, siluOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnSilu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSilu failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul5Output);

    // rmsnorm2Output -> matmul6Output
    std::vector<int64_t> matmul6WeightShape = {4096,11008};
    std::vector<int64_t> matmul6OutputShape = {1, maxInputLegth, 11008};
    aclTensor* matmul6Weight = nullptr;
    aclTensor* matmul6Output = nullptr;
    int8_t matmul6CubeMathType=0;
    ret = CreateAclTensorWithData(matmul6WeightShape, &decoderLayerInfo.matmul6WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul6Weight);
    ret = CreateAclTensorWithData(matmul6OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul6Output);
    ret = aclnnMatmulGetWorkspaceSize(rmsnorm2Output, matmul6Weight, matmul6Output, matmul6CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul6Weight);
    aclDestroyTensor(rmsnorm2Output);

    // matmul6Output * siluOutput -> mulOutput
    std::vector<int64_t> mulOutputShape = {1, maxInputLegth, 11008};
    aclTensor* mulOutput = nullptr;
    ret = CreateAclTensorWithData(mulOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &mulOutput);
    
    ret = aclnnMulGetWorkspaceSize(matmul6Output, siluOutput, mulOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul6Output);
    aclDestroyTensor(siluOutput);
    // mulOutput -> matmul7Output
    std::vector<int64_t> matmul7WeightShape = {11008, 4096};
    std::vector<int64_t> matmul7OutputShape = {1, maxInputLegth, 4096};
    aclTensor* matmul7Weight = nullptr;
    aclTensor* matmul7Output = nullptr;
    
    int8_t matmul7CubeMathType=0;
    ret = CreateAclTensorWithData(matmul7WeightShape, &decoderLayerInfo.matmul7WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul7Weight);
    ret = CreateAclTensorWithData(matmul7OutputShape, &maxDev_c, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmul7Output);
    ret = aclnnMatmulGetWorkspaceSize(mulOutput, matmul7Weight, matmul7Output, matmul7CubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul7Weight);
    aclDestroyTensor(mulOutput);
    // matmul7Output + add1Output-> add2Output
    aclScalar* add2Alpha = aclCreateScalar(commonInfo.alphaDev, aclDataType::ACL_FLOAT16);
    std::vector<int64_t> add2OutputShape = {1, maxInputLegth, 4096};
    aclTensor* add2Output = nullptr;
    ret = CreateAclTensorWithData(add2OutputShape, decoderLayerOutputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &add2Output);
    ret = aclnnAddGetWorkspaceSize(matmul7Output, add1Output, add2Alpha, add2Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdd failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmul7Output);
    aclDestroyTensor(add1Output);
    aclDestroyScalar(add2Alpha);
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    aclrtFree(maxDev_c);
    maxDev_c = nullptr;
    aclrtFree(maxDev_d);
    maxDev_d = nullptr;
    return ACL_SUCCESS;
}

aclError ExcuteLlama1st(LlamaInfo& llamaInfo, void** llama1stOutDev, aclrtStream& stream) {
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;
    // malloc dev mem for infer
    auto maxDevSize = maxInputLegth * 32001 * sizeof(float);
    void* maxDev_a = nullptr; 
    void* maxDev_b = nullptr;
    auto ret = aclrtMalloc(&maxDev_a, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMalloc(&maxDev_b, maxDevSize, ACL_MEM_MALLOC_NORMAL_ONLY);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // head
    // inputIds + embedTokens -> gatherOutput
    std::vector<int64_t> gatherWeightShape = {32001, 4096};
    std::vector<int64_t> inputIdsShape = {1, maxInputLegth};
    std::vector<int64_t> gatherOutputShape = {1, maxInputLegth, 4096};
    aclTensor* gatherWeight = nullptr;
    aclTensor* inputIds = nullptr;
    aclTensor* gatherOutput = nullptr;
    ret = CreateAclTensorWithData(gatherWeightShape, &llamaInfo.embedTokensWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &gatherWeight);
    ret = CreateAclTensorWithData(inputIdsShape, &llamaInfo.inputIdsDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &inputIds);
    ret = CreateAclTensorWithData(gatherOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &gatherOutput);
    int64_t sinDim = 0;
    ret = aclnnGatherV2GetWorkspaceSize(gatherWeight, sinDim, inputIds, gatherOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(gatherWeight);
    aclDestroyTensor(inputIds);
    for(int i=0; i < 32; i++) {
        ret = ExcuteDecoderLayer(gatherOutput, &maxDev_a, llamaInfo.decoderLayerInfo[i], stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteDecoderLayer failed. ERROR: %d\n", ret); return ret);
    }
    // tail
    // gatherOutput - > rmsnormOutput
    std::vector<int64_t> rmsnormOutputShape = {1, maxInputLegth, 4096};
    aclTensor* rmsnormOutput = nullptr;
    ret = CreateAclTensorWithData(rmsnormOutputShape, &maxDev_b, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &rmsnormOutput);    
    ret = ExcuteRmsnorm(gatherOutput, &maxDev_b, llamaInfo.lastRmsWeightDev, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExcuteRmsnorm failed. ERROR: %d\n", ret); return ret);
    aclDestroyTensor(gatherOutput);

    // rmsnormOutput - > matmulOutput
    std::vector<int64_t> matmulWeightShape = {4096, 32001};
    std::vector<int64_t> matmulOutputShape = {1, maxInputLegth, 32001};
    aclTensor* matmulWeight = nullptr;
    aclTensor* matmulOutput = nullptr;
    int8_t matmulCubeMathType=0;
    ret = CreateAclTensorWithData(matmulWeightShape, &llamaInfo.lmHeadWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmulWeight);
    ret = CreateAclTensorWithData(matmulOutputShape, &maxDev_a, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND, &matmulOutput);
    ret = aclnnMatmulGetWorkspaceSize(rmsnormOutput, matmulWeight, matmulOutput, matmulCubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnMatmul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(rmsnormOutput);
    aclDestroyTensor(matmulWeight);

    // matmulOutput - > castOutput
    std::vector<int64_t> castOutputShape = {1, maxInputLegth, 32001};
    aclTensor* castOutput = nullptr;
    ret = CreateAclTensorWithData(castOutputShape, &maxDev_b, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &castOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclnnCastGetWorkspaceSize(matmulOutput, aclDataType::ACL_FLOAT, castOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(matmulOutput);
    
    // castOutput - > gather1Output
    std::vector<int64_t> gather1IndexShape = {1};
    std::vector<int64_t> gather1OutputShape = {1, 1, 32001};
    void* gather1IndexDev = nullptr;
    aclTensor* gather1Index = nullptr;
    aclTensor* gather1Output = nullptr;
    int64_t gather1IndexData = 15;
    ret = CreateAclTensor(gather1IndexData, gather1IndexShape, &gather1IndexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &gather1Index);
    ret = CreateAclTensorWithData(gather1OutputShape, &maxDev_a, aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND, &gather1Output);
    int64_t gather1Dim = 1;
    ret = aclnnGatherV2GetWorkspaceSize(castOutput, gather1Dim, gather1Index, gather1Output, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGatherGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnGatherV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGather failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(castOutput);
    aclDestroyTensor(gather1Index);
    aclrtFree(gather1IndexDev);
    gather1IndexDev = nullptr;

    // gather1Output - > argMaxOutput
    std::vector<int64_t> argMaxOutputShape = {1, 1, 1};
    void* argMaxOutputDev = nullptr;
    aclTensor* argMaxOutput = nullptr;
    ret = CreateAclTensor(argMaxOutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &argMaxOutput);
    int64_t argMaxDim = -1;
    bool argMaxKeepdim = true;
    ret = aclnnArgMaxGetWorkspaceSize(gather1Output, argMaxDim, argMaxKeepdim, argMaxOutput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnArgMax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnArgMax failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(gather1Output);
    
    // argMaxOutput(reshape1Input) - > reshape1Output
    std::vector<int64_t> reshape1OutputShape = {1, 1};
    aclTensor* reshape1Input = nullptr;
    aclTensor* reshape1Output = nullptr;
    ret = CreateAclTensorWithData(reshape1OutputShape, &argMaxOutputDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &reshape1Input);
    ret = CreateAclTensorWithData(reshape1OutputShape, llama1stOutDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &reshape1Output);
    ret = aclnnInplaceCopyGetWorkspaceSize(reshape1Output, reshape1Input, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
        workspaceSize = 0;
        workspaceAddr = nullptr;
    }
    aclDestroyTensor(reshape1Input);
    aclDestroyTensor(argMaxOutput);
    aclDestroyTensor(reshape1Output);
    aclrtFree(argMaxOutputDev);
    argMaxOutputDev = nullptr;
    // release dev mem for infer
    aclrtFree(maxDev_a);
    maxDev_a = nullptr;
    aclrtFree(maxDev_b);
    maxDev_b = nullptr;
    return ACL_SUCCESS;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // init acl resource
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateContext(context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetCurrentContext(*context);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  aclrtRunMode runMode;
  aclrtGetRunMode(&runMode);
  g_isDevice = (runMode == ACL_DEVICE);
  return 0;
}

int main() {
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::cout<<"[INFO] data preparing..."<<std::endl;
    // prepare input data & weight
    LlamaInfo llamaInfo;
    std::vector<int64_t> embedTokensWeightShape = {32001, 4096};
    std::string embedTokensWeightPath = "../llama_weight/model.embed_tokens.weight.bin";
    ret = ReadDataToDevice(embedTokensWeightPath, embedTokensWeightShape, &llamaInfo.embedTokensWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
    // kv max addr malloc, shape = maxInputLegth + maxInferNum
    std::vector<int64_t> stateMemShape = {1, 32, maxInputLegth + maxInferNum, 128};
    size_t stateMemSize = GetShapeSize(stateMemShape) * sizeof(uint16_t);
    void* keyStateMem[32];
    void* valueStateMem[32];
    void *alphaPtr = nullptr;
    ret = aclrtMallocHost(&alphaPtr, sizeof(uint16_t));
    size_t fileSize = 0;
    ReadFile("../data/alpha_ones.bin", fileSize, reinterpret_cast<void *>(alphaPtr), sizeof(uint16_t));

    void* sinInputDev = nullptr;
    void* cosInputDev = nullptr;
    std::vector<int64_t> inputShape = {2048, 128};
    std::string sinInputPath = "../data/aclnngather_X_sin_add.bin";
    std::string cosInputPath = "../data/aclnngather_X_cos_add.bin";
    ret = ReadDataToDevice(sinInputPath, inputShape, &sinInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
    ret = ReadDataToDevice(cosInputPath, inputShape, &cosInputDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);

    void* attenMaskDev = nullptr;
    std::vector<int64_t> attenMaskShape = {2048, 2048};
    std::string attenMaskPath = "../data/attention_mask_for_aclnnPromptFlashAttention.bin";
    ret = ReadDataToDevice(attenMaskPath, attenMaskShape, &attenMaskDev, aclDataType::ACL_UINT8, aclFormat::ACL_FORMAT_ND);
    commonInfo.alphaDev = alphaPtr;
    commonInfo.attenMaskDev = attenMaskDev;
    commonInfo.sinInputDev = sinInputDev;
    commonInfo.cosInputDev = cosInputDev;
    for (int i = 0; i < 32; i++) {
        DecoderLayerInfo dlInfo;
        ret = aclrtMalloc(&keyStateMem[i], stateMemSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == 0, LOG_PRINT("acldvppMalloc for keyStateMem failed. ERROR: %d\n", ret); return ret);
        dlInfo.keyStateMemList = &keyStateMem[i];
        ret = aclrtMalloc(&valueStateMem[i], stateMemSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == 0, LOG_PRINT("acldvppMalloc for valueStateMem failed. ERROR: %d\n", ret); return ret);
        dlInfo.valueStateMemList = &valueStateMem[i];

        std::string head = "../llama_weight/model.layers." + INT2STR[i];
        std::vector<int64_t> matmul1WeightShape = {4096, 4096};
        std::string matmul1WeightPath = head + ".self_attn.q_proj.weight.bin";
        ret = ReadDataToDevice(matmul1WeightPath, matmul1WeightShape, &dlInfo.matmul1WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul2WeightShape = {4096, 4096};
        std::string matmul2WeightPath = head + ".self_attn.k_proj.weight.bin";
        ret = ReadDataToDevice(matmul2WeightPath, matmul2WeightShape, &dlInfo.matmul2WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul3WeightShape = {4096, 4096};
        std::string matmul3WeightPath = head + ".self_attn.v_proj.weight.bin";
        ret = ReadDataToDevice(matmul3WeightPath, matmul3WeightShape, &dlInfo.matmul3WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul4WeightShape = {4096, 4096};
        std::string matmul4WeightPath = head + ".self_attn.o_proj.weight.bin";
        ret = ReadDataToDevice(matmul4WeightPath, matmul4WeightShape, &dlInfo.matmul4WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul5WeightShape = {4096, 11008};
        std::string matmul5WeightPath = head + ".mlp.gate_proj.weight.bin";
        ret = ReadDataToDevice(matmul5WeightPath, matmul5WeightShape, &dlInfo.matmul5WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul6WeightShape = {4096, 11008};
        std::string matmul6WeightPath = head + ".mlp.up_proj.weight.bin";
        ret = ReadDataToDevice(matmul6WeightPath, matmul6WeightShape, &dlInfo.matmul6WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> matmul7WeightShape = {11008, 4096};
        std::string matmul7WeightPath = head + ".mlp.down_proj.weight.bin";
        ret = ReadDataToDevice(matmul7WeightPath, matmul7WeightShape, &dlInfo.matmul7WeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::vector<int64_t> mulWeightShape = {1, 1, 4096};
        std::string rm1mulWeightPath = head + ".input_layernorm.weight.bin";
        ret = ReadDataToDevice(rm1mulWeightPath, mulWeightShape, &dlInfo.headRmsWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        std::string rm2mulWeightPath = head + ".post_attention_layernorm.weight.bin";
        ret = ReadDataToDevice(rm2mulWeightPath, mulWeightShape, &dlInfo.tailRmsWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
        llamaInfo.decoderLayerInfo.push_back(dlInfo);
    }
    std::string rmmulWeightPath = "../llama_weight/model.norm.weight.bin";
    std::vector<int64_t> mulWeightShape = {1, 1, 4096};
    ret = ReadDataToDevice(rmmulWeightPath, mulWeightShape, &llamaInfo.lastRmsWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
    std::string lmHeadWeight = "../llama_weight/lm_head.weight.bin";
    std::vector<int64_t> lmHeadWeightShape = {4096, 32001};
    ret = ReadDataToDevice(lmHeadWeight, lmHeadWeightShape, &llamaInfo.lmHeadWeightDev, aclDataType::ACL_FLOAT16, aclFormat::ACL_FORMAT_ND);
    std::cout<<"[INFO] data prepared, ready to infer llama."<<std::endl;

    auto inferOutSize = sizeof(int64_t);
    void* inferOutHostData = nullptr;
    ret = aclrtMallocHost(&inferOutHostData, inferOutSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost host failed. ERROR: %d\n", ret); return ret);

    int i = 0;
    std::stringstream sstream;
    while(1) {
        // If the file name is exit.bin, exit the loop
        struct stat buffer;
        auto exit_value = stat("../input/exit.bin", &buffer);
        if(! exit_value){
            break;
        }

        std::string inputIdsPath;
        sstream.str("");
        sstream.clear();
        sstream << "../input/input_ids_" << i << ".bin";
        sstream >> inputIdsPath;
        exit_value = stat(inputIdsPath.c_str(), &buffer);
        if(exit_value){
           sleep(1);
           continue; 
        }
        // get real input sequence length, update maxInputLegth
        size_t fileSize = 0;
        auto retRead = ReadFile("../input/maxInputLegth.bin", fileSize, &maxInputLegth, sizeof(int64_t));
        CHECK_RET(retRead == true, LOG_PRINT("ReadFile maxInputLegth failed. ERROR: %d\n", retRead); return !retRead);

        std::vector<int64_t> inputIdsShape = {1, maxInputLegth};
        ret = ReadDataToDevice(inputIdsPath, inputIdsShape, &llamaInfo.inputIdsDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND);
        llamaInfo.posIndex = maxInputLegth;
        
        void* indexDev = nullptr;
        std::vector<int64_t> indexShape = {1, maxInputLegth};
        std::string indexPath = "../input/aclnngather_index.bin";

        ret = ReadDataToDevice(indexPath, indexShape, &indexDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND);
        commonInfo.indexDev = indexDev;

        std::vector<int64_t> llama1stOutShape = {1, 1};
        void* llama1stOutDev = nullptr;
        aclTensor* llama1stOut = nullptr;
        ret = CreateAclTensor(llama1stOutShape, &llama1stOutDev, aclDataType::ACL_INT64, aclFormat::ACL_FORMAT_ND, &llama1stOut);
        ExcuteLlama1st(llamaInfo, &llama1stOutDev, stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] llama 1st run failed. ERROR: %d\n", ret); return ret);
        std::cout<<"[INFO] llama 1st run SUCCESS"<<std::endl;

        std::vector<int64_t> answerList;
        ret = aclrtMemcpy(inferOutHostData, inferOutSize, llama1stOutDev, inferOutSize, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
        int64_t inferOut = *(reinterpret_cast<int64_t*>(inferOutHostData));
        answerList.push_back(inferOut);
        // ExcuteLlama2nd
        int endcnt=3;
        for(int i=0; (i < maxInferNum) && endcnt; i++) {
            ret = ExcuteLlama2nd(llama1stOut, &llama1stOutDev, llamaInfo, stream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] llama 2nd run failed. ERROR: %d\n", ret); return ret);
            llamaInfo.posIndex++;
            std::cout<<"[INFO] llama 2nd run "<< i <<" SUCCESS"<<std::endl;
            ret = aclrtMemcpy(inferOutHostData, inferOutSize, llama1stOutDev, inferOutSize, ACL_MEMCPY_DEVICE_TO_HOST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
            inferOut = *(reinterpret_cast<int64_t*>(inferOutHostData));
            answerList.push_back(inferOut);
            if (inferOut==13) {
                endcnt--;
            }
        }
        // release resource
        aclDestroyTensor(llama1stOut);
        aclrtFree(llama1stOutDev);
        llama1stOutDev = nullptr;

        std::string outputPath = "";
        sstream.str("");
        sstream.clear();
        sstream << "../output/fin_result_" << i << ".bin";
        sstream >> outputPath;
        WriteFile(outputPath, answerList.data(), inferOutSize*answerList.size());
        i++;
    }
    return 0;
}