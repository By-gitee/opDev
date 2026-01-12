#include "sparse_mat_mul_t_base_csr.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseMatMulTBaseCSRInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseMatMulTBaseCSR, SparseMatMulTBaseCSRVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseMatMulTBaseCSR, SparseMatMulTBaseCSRInferShape);
VERIFY_FUNC_REG(SparseMatMulTBaseCSR, SparseMatMulTBaseCSRVerify);

}  // namespace ge
