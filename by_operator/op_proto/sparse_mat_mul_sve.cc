#include "sparse_mat_mul_sve.h"
namespace ge {
bool InferShapeAndTypeSparseMatMulTBaseSVE(Operator& op, const string& input_name1, const string& input_name2, const string& output_name) {

 TensorDesc vOutputDesc = op.GetOutputDescByName(output_name.c_str());

  const TensorDesc Input1Desc = op.GetInputDescByName(input_name1.c_str());
  const TensorDesc Input2Desc = op.GetInputDescByName(input_name2.c_str());

  const DataType Input1Dtype = Input1Desc.GetDataType();
  const Format Input1Format = Input1Desc.GetFormat();

  // 1) Check dtype in Verify func

  // 2) Get shape
  ge::Shape Input1Shape = Input1Desc.GetShape();
  ge::Shape Input2Shape = Input2Desc.GetShape();
  std::vector<int64_t> Input1Dims = Input1Shape.GetDims();
  std::vector<int64_t> Input2Dims = Input2Shape.GetDims();

  // MatMul requires at least 2D inputs
  if (Input1Dims.size() < 2 || Input2Dims.size() < 2) {
    return false;
  }

  const size_t Input1Rank = Input1Dims.size();
  const size_t Input2Rank = Input2Dims.size();

  // 3) Get matrix dims：Input1[..., M, K], Input2[..., N, K] (already transposed)
  int64_t Input1M = Input1Dims[Input1Rank - 2];
  // int64_t Input1K = Input1Dims[Input1Rank - 1];
  int64_t Input2N = Input2Dims[Input2Rank - 2];  // Input2 is already transposed, so N is at index -2
  // int64_t Input2K = Input2Dims[Input2Rank - 1];  // Input2 is already transposed, so K is at index -1

  // 4) Check K dim compatibility (K dimension should match between Input1[..., M, K] and Input2[..., N, K])
  // auto is_known = [](int64_t d) { return d >= 0; }; // 0 is treated as known here(empty dim)
  // if (is_known(Input1K) && is_known(Input2K) && Input1K != Input2K) {
    // return false;
  // }

  // 5) Handle batch dimension broadcasting
  // batchInput1 = Input1Dims[0 : Input1Rank-2]
  // batchInput2 = Input2Dims[0 : Input2Rank-2]
  std::vector<int64_t> b1(Input1Dims.begin(), Input1Dims.end() - 2);
  std::vector<int64_t> b2(Input2Dims.begin(), Input2Dims.end() - 2);

  if (b1.size() < b2.size()) {
    b1.insert(b1.begin(), b2.size() - b1.size(), (int64_t)1);
  } else if (b2.size() < b1.size()) {
    b2.insert(b2.begin(), b1.size() - b2.size(), (int64_t)1);
  }

  std::vector<int64_t> bout;
  bout.reserve(b1.size());

  for (size_t i = 0; i < b1.size(); ++i) {
    int64_t a = b1[i];
    int64_t b = b2[i];

    // Propagate empty dimension:
    // if either dimension is 0, output dimension is 0
    if (a == 0 || b == 0) {
      bout.push_back(0);
      continue;
    }

    // Known but incompatible dimensions
    // (neither is 1 nor -1)
    if (a != b && a != 1 && b != 1 && a != -1 && b != -1) {
      return false;
    }

    // Compatible dimension inference:
    // - If one dimension > 1, prefer that one
    // - If one is -1 (unknown) and the other is 1, result remains unknown (-1)
    // - If one is -1 and the other > 1, result can be inferred as > 1
    int64_t out_d = 1;
    if (a == -1 && b == -1) {
      out_d = -1;
    } else if (a == -1) {
      out_d = (b == 1) ? -1 : b;
    } else if (b == -1) {
      out_d = (a == 1) ? -1 : a;
    } else {
      // Both dimensions are known and compatible
      out_d = (a > b) ? a : b; // 1 vs N -> N, N vs N -> N
    }
    bout.push_back(out_d);
  }

  // 6) Compose output shape: batch_dims + [M, N] (result of multiplying Input1[..., M, K] with Input2[..., N, K] where Input2 is already transposed)
  std::vector<int64_t> OutDims = bout;
  OutDims.push_back(Input1M);
  OutDims.push_back(Input2N);

  ge::Shape OutShape(OutDims);

  vOutputDesc.SetShape(OutShape);
  vOutputDesc.SetDataType(Input1Dtype);
  vOutputDesc.SetFormat(Input1Format);
  op.UpdateOutputDesc(output_name.c_str(), vOutputDesc);

  return true;
}

IMPLEMT_COMMON_INFERFUNC(SparseMatMulTBaseSVEInferShape)
{
    if(InferShapeAndTypeSparseMatMulTBaseSVE(op, "sparse_matrix", "weight", "output")) {
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

IMPLEMT_VERIFIER(SparseMatMulTBaseSVE, SparseMatMulTBaseSVEVerify)
{
  DataType Input1Type = op.GetInputDescByName("sparse_matrix").GetDataType();
  DataType Input2Type = op.GetInputDescByName("weight").GetDataType();
  if (Input1Type != Input2Type) {
    return GRAPH_FAILED;
  }
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(SparseMatMulTBaseSVE, SparseMatMulTBaseSVEInferShape);
VERIFY_FUNC_REG(SparseMatMulTBaseSVE, SparseMatMulTBaseSVEVerify);

}  // namespace ge
