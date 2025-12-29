#include "aclnn/acl_meta.h"
#include "common.h"

namespace AclnnLlama{
class CommonInfo {
public:
    void* alphaDev = nullptr;
    void* attenMaskDev = nullptr;
    void* sinInputDev = nullptr;
    void* cosInputDev = nullptr;
    void* indexDev = nullptr;
};
class DecoderLayerInfo {
public:
    void* matmul1WeightDev = nullptr;
    void* matmul2WeightDev = nullptr;
    void* matmul3WeightDev = nullptr;
    void* matmul4WeightDev = nullptr;
    void* matmul5WeightDev = nullptr;
    void* matmul6WeightDev = nullptr;
    void* matmul7WeightDev = nullptr;
    void* headRmsWeightDev = nullptr;
    void* tailRmsWeightDev = nullptr;
    void** keyStateMemList;
    void** valueStateMemList;
};
class LlamaInfo {
public:
    std::vector<DecoderLayerInfo> decoderLayerInfo;
    void* lastRmsWeightDev = nullptr;
    void* inputIdsDev = nullptr;
    void* embedTokensWeightDev = nullptr;
    void* lmHeadWeightDev = nullptr;
    int64_t posIndex;
};
} // AclnnLlama
