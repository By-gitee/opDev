#include "dense_to_csr.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(DenseToCSRInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DenseToCSR, DenseToCSRVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DenseToCSR, DenseToCSRInferShape);
VERIFY_FUNC_REG(DenseToCSR, DenseToCSRVerify);

}  // namespace ge
