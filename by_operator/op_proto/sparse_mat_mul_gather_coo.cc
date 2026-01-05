#include "sparse_mat_mul_gather_coo.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(SparseMatMulGatherCOOInferShape)
{
    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(SparseMatMulGatherCOO, SparseMatMulGatherCOOVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseMatMulGatherCOO, SparseMatMulGatherCOOInferShape);
VERIFY_FUNC_REG(SparseMatMulGatherCOO, SparseMatMulGatherCOOVerify);

}  // namespace ge
