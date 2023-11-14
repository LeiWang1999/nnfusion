#include "cutlass/cutlass.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/mma_tensor_op_sm70.h"

namespace cutlass {
namespace gemm {
namespace warp {

template <
  typename Shape,
  typename SMemLayoutA,
  typename SMemLayoutB
>
class GemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 8, 16>;

  using WarpShape = GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::ColumnMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaTensorOp<
    WarpShape,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    Policy
  >;
  static_assert(Shape::kK % InstructionShape::kK == 0);
  static int const kKgroups = Shape::kK / InstructionShape::kK;

  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;

  typename MmaWarp::FragmentA frag_A[2];
  typename MmaWarp::FragmentB frag_B[2];
  typename MmaWarp::FragmentC accum;
  MmaWarp mma_op;
  typename MmaWarp::IteratorA iter_A;
  typename MmaWarp::IteratorB iter_B;
  const int warp_idx_m_, warp_idx_n_, lane_id_;
public:
  CUTLASS_DEVICE
  GemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
    accum.clear();
  }
  CUTLASS_DEVICE
  void prologue(const TensorRefA &ref_A, const TensorRefB &ref_B) {
    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
    iter_A.add_tile_offset({warp_idx_m_, 0});
    iter_B.add_tile_offset({0, warp_idx_n_});
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);
    ++iter_A;
    ++iter_B;
  }
  CUTLASS_DEVICE
  void body() {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups - 1; ++k) {
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
    __syncthreads();
  }
  CUTLASS_DEVICE
  void epilogue() {
    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
  }
  CUTLASS_DEVICE
  half& operator[](const size_t i) const {
    return ((half*)accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(const size_t i) const {
    return (half*)accum.data() + i;
  }
};

template <
  typename Shape,
  typename SMemLayoutA,
  typename LayoutA,
  typename SMemLayoutB,
  typename LayoutB,
  typename LayoutC
>
class VoltaGemmTensorOp {
public:
  using InstructionShape = GemmShape<16, 16, 4>;
  using WarpShape = GemmShape<Shape::kM, Shape::kN, InstructionShape::kK>;
  using Policy = cutlass::gemm::warp::MmaTensorOpPolicy<
    cutlass::arch::Mma<
      InstructionShape,
      32,
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      LayoutC,
      cutlass::arch::OpMultiplyAdd
    >,
    cutlass::MatrixShape<1, 1>
  >;

  using MmaWarp = typename cutlass::gemm::warp::MmaVoltaTensorOp<
    WarpShape,
    cutlass::half_t,
    SMemLayoutA,
    cutlass::half_t,
    SMemLayoutB,
    cutlass::half_t,
    LayoutC,
    Policy
  >;
  static_assert(Shape::kK % InstructionShape::kK == 0);
  static int const kKgroups = Shape::kK / InstructionShape::kK;
  using TensorRefA = typename MmaWarp::IteratorA::TensorRef;
  using TensorRefB = typename MmaWarp::IteratorB::TensorRef;

  typename MmaWarp::FragmentA frag_A[2];
  typename MmaWarp::FragmentB frag_B[2];
  typename MmaWarp::FragmentC accum;
  typename MmaWarp::IteratorA iter_A;
  typename MmaWarp::IteratorB iter_B;

  const int warp_idx_m_, warp_idx_n_, lane_id_;
  MmaWarp mma_op;
public:
  CUTLASS_DEVICE
  VoltaGemmTensorOp(int warp_idx_m, int warp_idx_n, int lane_id)
  : warp_idx_m_(warp_idx_m), warp_idx_n_(warp_idx_n), lane_id_(lane_id), iter_A({nullptr, 0}, 0), iter_B({nullptr, 0}, 0) {
    accum.clear();
  }
  CUTLASS_DEVICE
  void prologue(TensorRefA ref_A, TensorRefB ref_B) {
    iter_A = typename MmaWarp::IteratorA(ref_A, lane_id_);
    iter_B = typename MmaWarp::IteratorB(ref_B, lane_id_);
    iter_A.add_tile_offset({warp_idx_m_, 0});
    iter_B.add_tile_offset({0, warp_idx_n_});
    iter_A.load(frag_A[0]);
    iter_B.load(frag_B[0]);
    ++iter_A;
    ++iter_B;
  }
  CUTLASS_DEVICE
  void body() {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < kKgroups - 1; ++k) {
      iter_A.load(frag_A[(k + 1) % 2]);
      iter_B.load(frag_B[(k + 1) % 2]);
      ++iter_A;
      ++iter_B;
      mma_op(accum, frag_A[k % 2], frag_B[k % 2], accum);
    }
    __syncthreads();
  }
  CUTLASS_DEVICE
  void epilogue() {
    mma_op(accum, frag_A[(kKgroups - 1) % 2], frag_B[(kKgroups - 1) % 2], accum);
  }
  CUTLASS_DEVICE
  half& operator[](size_t i) const {
    return ((half*)accum.data())[i];
  }
  CUTLASS_DEVICE
  half* operator+(size_t i) const {
    return (half*)accum.data() + i;
  }
};

}}}

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY
