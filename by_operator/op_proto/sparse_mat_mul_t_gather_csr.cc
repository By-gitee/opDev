#include "sparse_mat_mul_t_gather_csr.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseMatMulTGatherCSRInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseMatMulTGatherCSR, SparseMatMulTGatherCSRVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseMatMulTGatherCSR, SparseMatMulTGatherCSRInferShape);
VERIFY_FUNC_REG(SparseMatMulTGatherCSR, SparseMatMulTGatherCSRVerify);

}  // namespace ge
