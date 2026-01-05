#include "dense_to_coo.h"
namespace ge {

IMPLEMT_COMMON_INFERFUNC(DenseToCOOInferShape)
{
    std::vector<int64_t> out_dims = { ge::UNKNOWN_DIM };

    TensorDesc out_x = op.GetOutputDesc("indicesX");
    out_x.SetShape(Shape(out_dims));
    out_x.SetDataType(DT_INT32);
    op.UpdateOutputDesc("indicesX", out_x);

    TensorDesc out_y = op.GetOutputDesc("indicesY");
    out_y.SetShape(Shape(out_dims));
    out_y.SetDataType(DT_INT32);
    op.UpdateOutputDesc("indicesY", out_y);

    const TensorDesc& thr_desc = op.GetInputDesc("threshold");
    const Shape& thr_shape = thr_desc.GetShape();
    if (thr_shape.GetDims() != UNKNOWN_RANK) {
        if(thr_shape.GetDimNum()!=0 && !(thr_shape.GetDimNum()==1 && thr_shape.GetDim(0)==1)) {
            return GRAPH_FAILED;
        }
    }

    return GRAPH_SUCCESS;
}

IMPLEMT_VERIFIER(DenseToCOO, DenseToCOOVerify)
{
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(DenseToCOO, DenseToCOOInferShape);
VERIFY_FUNC_REG(DenseToCOO, DenseToCOOVerify);

}  // namespace ge
