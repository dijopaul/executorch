/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include"kernels.h"

#define NNLIB_OPT 0

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

namespace {
template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void run(const Tensor& a, const Tensor& b, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = std::pow(a_casted, b_casted);
          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct PowInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug {};

} // namespace

Tensor& pow_Tensor_Tensor_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
#if NNLIB_OPT
    
  const bool a_is_broadcasted = !out.sizes().equals(a.sizes());
  const bool b_is_broadcasted = !out.sizes().equals(b.sizes());
  const bool both_is_broadcasted = (a_is_broadcasted && b_is_broadcasted);

  WORD32 num_elm = out.numel();
  
  if(both_is_broadcasted)
  {
    WORD32* __restrict__ ptr1 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
    WORD32* __restrict__ ptr2 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
        
    WORD32* __restrict__ pin1 = (WORD32* __restrict__)a.const_data_ptr<float>();
    WORD32* __restrict__ pin2 = (WORD32* __restrict__)b.const_data_ptr<float>();
    
    WORD32 p_out_shape[4];
    WORD32 p_inp1_shape[4];
    WORD32 p_inp2_shape[4];
    
    for(int i = 0; i < 4; i++)
    {
      p_inp1_shape[i] = 1;
      p_inp2_shape[i] = 1;
      p_out_shape[i] = 1;
    }
      
    int off_o = 4 - out.dim();        
    int off_a = 4 - a.dim();
    int off_b = 4 - b.dim();

    for(int i = 0; i < out.dim(); i++)
      p_out_shape[i+off_o] = out.size(i);
    for(int i = 0; i < a.dim(); i++)
      p_inp1_shape[i+off_a] = a.size(i);
    for(int i = 0; i < b.dim(); i++)
      p_inp2_shape[i+off_b] = b.size(i);

    WORD32 val = xa_nn_broadcast_32_32(ptr1,      
                            p_out_shape,             
                            pin1,        
                            p_inp1_shape,             
                            out.dim());
                          
    val = xa_nn_broadcast_32_32(ptr2,      
                            p_out_shape,             
                            pin2,        
                            p_inp2_shape,             
                            out.dim());
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)ptr1;
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)ptr2;
    
    vecpowf (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm ); 
    
    free(ptr1);
    free(ptr2);
  }
  else if(a_is_broadcasted && (!b_is_broadcasted))
  {
    FLOAT32* __restrict__ ptr1 = (FLOAT32* __restrict__ )malloc((num_elm + 2) * sizeof(WORD32));
        
    FLOAT32* __restrict__ pin1 = (FLOAT32* __restrict__)a.const_data_ptr<float>();
    
    WORD32 p_out_shape[4];
    WORD32 p_inp1_shape[4];
    
    for(int i = 0; i < 4; i++)
    {
      p_inp1_shape[i] = 1;
      p_out_shape[i] = 1;
    }
      
    int off_o = 4 - out.dim();        
    int off_a = 4 - a.dim();

    for(int i = 0; i < out.dim(); i++)
      p_out_shape[i+off_o] = out.size(i);
    for(int i = 0; i < a.dim(); i++)
      p_inp1_shape[i+off_a] = a.size(i);

    WORD32 val = xa_nn_broadcast_32_32((WORD32 *)ptr1,      
                            p_out_shape,             
                            (WORD32 *)pin1,        
                            p_inp1_shape,             
                            4);
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)ptr1;
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
    
    vecpowf (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm); 
    
    free(ptr1);
  }
  else if(b_is_broadcasted && (!a_is_broadcasted))
  {
    WORD32* __restrict__ ptr1 = (WORD32* __restrict__ )malloc(num_elm * sizeof(WORD32));
        
    WORD32* __restrict__ pin1 = (WORD32* __restrict__)b.const_data_ptr<float>();
    
    WORD32 p_out_shape[4];
    WORD32 p_inp1_shape[4];
    
    for(int i = 0; i < 4; i++)
    {
      p_inp1_shape[i] = 1;
      p_out_shape[i] = 1;
    }
      
    int off_o = 4 - out.dim();        
    int off_a = 4 - b.dim();

    for(int i = 0; i < out.dim(); i++)
      p_out_shape[i+off_o] = out.size(i);
    for(int i = 0; i < a.dim(); i++)
      p_inp1_shape[i+off_a] = b.size(i);

    WORD32 val = xa_nn_broadcast_32_32(ptr1,      
                            p_out_shape,             
                            pin1,        
                            p_inp1_shape,             
                            out.dim());
            
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)ptr1;
    
    vecpowf (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm ); 
    
    free(ptr1);
  }
  else
  {
    FLOAT32 * __restrict__ p_out = (FLOAT32 * __restrict__ )out.mutable_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp1 = (const FLOAT32 * __restrict__)a.const_data_ptr<float>();
    const FLOAT32 * __restrict__ p_inp2 = (const FLOAT32 * __restrict__)b.const_data_ptr<float>();
    
    vecpowf (p_out, 
              p_inp1, 
              p_inp2, 
              num_elm );  
  }
#else    
  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(
      ctx, common_type != exec_aten::ScalarType::Bool, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "pow.Tensor_Tensor_out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(
        b_type, ctx, "pow.Tensor_Tensor_out", CTYPE_B, [&]() {
          using CTYPE_IN = typename torch::executor::
              promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
          ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
          ET_SWITCH_REALH_TYPES(
              out_type, ctx, "pow.Tensor_Tensor_out", CTYPE_OUT, [&]() {
                PowInner<
                    !std::is_same<CTYPE_IN, bool>::value &&
                        can_cast<CTYPE_IN, CTYPE_OUT>::value,
                    CTYPE_A,
                    CTYPE_B,
                    CTYPE_IN,
                    CTYPE_OUT>::run(a, b, out);
              });
        });
  });
#endif
  return out;
}

Tensor& pow_Tensor_Scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType common_type =
      utils::promote_type_with_scalar(a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_REALHB_TYPES(a_type, ctx, "pow.Tensor_Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_TYPES(
        b_type, ctx, "pow.Tensor_Scalar_out", CTYPE_B, [&]() {
          ET_SWITCH_REAL_TYPES(
              common_type, ctx, "pow.Tensor_Scalar_out", CTYPE_IN, [&]() {
                ET_SWITCH_REALH_TYPES(
                    out_type, ctx, "pow.Tensor_Scalar_out", CTYPE_OUT, [&]() {
                      CTYPE_B val_b = 0;
                      utils::extract_scalar(b, &val_b);
                      apply_unary_map_fn(
                          [val_b](const CTYPE_A val_a) {
                            CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                            CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                            CTYPE_IN value = std::pow(a_casted, b_casted);

                            return static_cast<CTYPE_OUT>(value);
                          },
                          a.const_data_ptr<CTYPE_A>(),
                          out.mutable_data_ptr<CTYPE_OUT>(),
                          out.numel());
                    });
              });
        });
  });

  return out;
}

Tensor& pow_Scalar_out(
    RuntimeContext& ctx,
    const Scalar& a,
    const Tensor& b,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, b.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ScalarType a_type = utils::get_scalar_dtype(a);
  ScalarType b_type = b.scalar_type();
  ScalarType common_type =
      utils::promote_type_with_scalar(b_type, a, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  ET_SWITCH_SCALAR_OBJ_TYPES(a_type, ctx, "pow.Scalar_out", CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, "pow.Scalar_out", CTYPE_B, [&]() {
      ET_SWITCH_REAL_TYPES(common_type, ctx, "pow.Scalar_out", CTYPE_IN, [&]() {
        ET_SWITCH_REALH_TYPES(
            out_type, ctx, "pow.Scalar_out", CTYPE_OUT, [&]() {
              CTYPE_A val_a = 0;
              utils::extract_scalar(a, &val_a);

              apply_unary_map_fn(
                  [val_a](const CTYPE_B val_b) {
                    CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
                    CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
                    CTYPE_IN value = std::pow(a_casted, b_casted);
                    return static_cast<CTYPE_OUT>(value);
                  },
                  b.const_data_ptr<CTYPE_B>(),
                  out.mutable_data_ptr<CTYPE_OUT>(),
                  out.numel());
            });
      });
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch