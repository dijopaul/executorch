/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>
#include "kernels.h"
#include <iostream>

#define NNLIB_OPT 0

namespace torch {
namespace executor {
namespace native {
namespace {
    


template <
    bool can_cast,
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct SubInner;

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct SubInner<true, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT> {
  static void
  run(const Tensor& a, const Tensor& b, CTYPE_IN alpha_val, Tensor& out) {
    apply_binary_elementwise_fn<CTYPE_A, CTYPE_B, CTYPE_OUT>(
        // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
        [alpha_val](const CTYPE_A val_a, const CTYPE_B val_b) {
          CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
          CTYPE_IN b_casted = static_cast<CTYPE_IN>(val_b);
          CTYPE_IN value = a_casted - alpha_val * b_casted;

          return static_cast<CTYPE_OUT>(value);
        },
        a,
        b,
        out);
  }
};

template <typename CTYPE_IN>
struct ReportCanCastBug {
  static void run(const Tensor&, const Tensor&, CTYPE_IN, Tensor&) {
    ET_DCHECK_MSG(false, "BUG: canCast should have been checked above");
  }
};

template <
    typename CTYPE_A,
    typename CTYPE_B,
    typename CTYPE_IN,
    typename CTYPE_OUT>
struct SubInner<false, CTYPE_A, CTYPE_B, CTYPE_IN, CTYPE_OUT>
    : public ReportCanCastBug<CTYPE_IN> {};

} // namespace

Tensor& sub_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Tensor& b,
    const Scalar& alpha,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, out) == Error::Ok,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(ctx, tensor_is_realh_type(out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);
  ScalarType common_type = promoteTypes(a_type, b_type, /*half_to_float*/ true);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, canCast(common_type, out_type), InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, check_alpha_type(alpha_type, common_type), InvalidArgument, out);

  constexpr auto name = "sub.out";
  
  //using CTYPE_IN = typename torch::executor::
    //      promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
  
  /*switch (alpha_type) {
    case ScalarType::Bool:
    {
        printf("Bool\n");
        break;
    }
    case ScalarType::Byte:
    {
         printf("Byte\n");
          break;
    }
    case ScalarType::Short:
    {
         printf("Short\n");
         break;
    }
    case ScalarType::Int:
    {
        printf("Int\n");
        break;
    }
    case ScalarType::Long:
    {
        printf("Long\n");
        break;
    }
    case ScalarType::Half:
    {
        printf("Half\n");
        break;
    }
    case ScalarType::Float:
    {
        printf("Float\n");
        break;
    }
    case ScalarType::Double:
    {
        printf("Double\n");
        break;
    }
    default:
    {
        printf("No MATCH\n");
        break;
    }
  }
  
  switch (common_type) {
    case ScalarType::Bool:
    {
        printf("Bool\n");
        break;
    }
    case ScalarType::Byte:
    {
         printf("Byte\n");
          break;
    }
    case ScalarType::Short:
    {
         printf("Short\n");
         break;
    }
    case ScalarType::Int:
    {
        printf("Int\n");
        break;
    }
    case ScalarType::Long:
    {
        printf("Long\n");
        break;
    }
    case ScalarType::Half:
    {
        printf("Half\n");
        break;
    }
    case ScalarType::Float:
    {
        printf("Float\n");
        break;
    }
    case ScalarType::Double:
    {
        printf("Double\n");
        break;
    }
    default:
    {
        printf("No MATCH\n");
        break;
    }
  }*/
  
  
  
  
  //float alpha_val;
  //utils::extract_scalar(alpha, &alpha_val);
  //printf("alpha = %f\n", alpha_val); 
  
  /*printf("*sub.out*\n");
  printf("a num = %d\n", a.numel());
  printf("b num  = %d\n", b.numel());
  printf("out num = %d\n", out.numel());
  
  printf("a dim = %zu\n", a.dim());
  printf("b dim = %zu\n", b.dim());
  printf("out dim = %zu\n", out.dim());*/
  
  /*printf("b size 0 = %zu\n", b.size(0));
  printf("b size 1 = %zu\n", b.size(1));*/
  
  /*int i;
  for(i = 0; i < out.dim(); i++)
      printf("out.size(%d):%d ", i, out.size(i));
  for(i = 0; i < a.dim(); i++)
      printf("a.size(%d):%d ", i, a.size(i));
  for(i = 0; i < b.dim(); i++)
      printf("b.size(%d):%d ", i, b.size(i));
  printf("\n");*/

  //printf("\n***************SUB***************");
#if NNLIB_OPT
#define NNLIB_MAX_DIM 4  /* Add fallback if broadcast and dim > 4 */
  if(out_type == ScalarType::Float)
  {
      /*logic to find broadcast*/
      const int a_is_broadcasted = !out.sizes().equals(a.sizes());
      const int b_is_broadcasted = !out.sizes().equals(b.sizes());
      const int broadcast = (a_is_broadcasted || b_is_broadcasted);
      
      const float* const a_data = a.const_data_ptr<float>();
      const float* const b_data = b.const_data_ptr<float>();
      float* const out_data = out.mutable_data_ptr<float>();
      if(broadcast == 1)
      {
         int out_shape[NNLIB_MAX_DIM];
         int inp1_shape[NNLIB_MAX_DIM];
         int inp2_shape[NNLIB_MAX_DIM];
         
         for(int i = 0; i < NNLIB_MAX_DIM; i++)
         {
            out_shape[i] = 1;
            inp1_shape[i] = 1;
            inp2_shape[i] = 1;
         }
         
         int a_dim = a.dim(), b_dim = b.dim(), out_dim = out.dim();
         
         
         
         int off_o = NNLIB_MAX_DIM - out_dim;
         int off_a = NNLIB_MAX_DIM - a_dim;
         int off_b = NNLIB_MAX_DIM - b_dim;
         for(int i = 0; i < out_dim; i++)
             out_shape[i+off_o] = out.size(i);
         for(int i = 0; i < a_dim; i++)
             inp1_shape[i+off_a] = a.size(i);
         for(int i = 0; i < b_dim; i++)
             inp2_shape[i+off_b] = b.size(i);

         xa_nn_elm_sub_broadcast_4D_f32xf32_f32(out_data, out_shape, a_data, inp1_shape,b_data, inp2_shape);
      }                      
      else
      {
         xa_nn_elm_sub_f32xf32_f32(out_data, a_data, b_data, out.numel());
      }

  }
  else
  {
  
     ET_SWITCH_REALH_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
      ET_SWITCH_REALH_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN = typename torch::executor::
          promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
      ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
      CTYPE_IN alpha_val;
      utils::extract_scalar(alpha, &alpha_val);
      ET_SWITCH_REALH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
        SubInner<
            can_cast<CTYPE_IN, CTYPE_OUT>::value,
            CTYPE_A,
            CTYPE_B,
            CTYPE_IN,
            CTYPE_OUT>::run(a, b, alpha_val, out);
         });
       });
     });
  }
#else
  
  ET_SWITCH_REALH_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALH_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN = typename torch::executor::
          promote_types<CTYPE_A, CTYPE_B, /*half_to_float*/ true>::type;
      ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);
      CTYPE_IN alpha_val;
      utils::extract_scalar(alpha, &alpha_val);
      ET_SWITCH_REALH_TYPES(out_type, ctx, name, CTYPE_OUT, [&]() {
        SubInner<
            can_cast<CTYPE_IN, CTYPE_OUT>::value,
            CTYPE_A,
            CTYPE_B,
            CTYPE_IN,
            CTYPE_OUT>::run(a, b, alpha_val, out);
      });
    });
  });
  

#endif  
  
  /*const float* const a_data = a.const_data_ptr<float>();
  printf("\n**** A *****\n");
  for(int i = 0; i < a.numel(); i++ )
    printf("%f \t", a_data[i]);
  const float* const b_data = b.const_data_ptr<float>();
  printf("\n**** B *****\n");
  for(int i = 0; i < b.numel(); i++ )
    printf("%f \t", b_data[i]);
  float* const out_data = out.mutable_data_ptr<float>();
  printf("\n**** out *****\n");
  for(int i = 0; i < out.numel(); i++ )
    printf("%f \t", out_data[i]);
  printf("\n");
  printf("\n***************END SUB***************");*/

  return out;
}

Tensor& sub_scalar_out(
    RuntimeContext& ctx,
    const Tensor& a,
    const Scalar& b,
    const Scalar& alpha,
    Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, a.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(ctx, tensor_is_realh_type(out), InvalidArgument, out);

  ScalarType a_type = a.scalar_type();
  ScalarType b_type = utils::get_scalar_dtype(b);
  ScalarType alpha_type = utils::get_scalar_dtype(alpha);
  ScalarType common_type =
      utils::promote_type_with_scalar(a_type, b, /*half_to_float*/ false);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, canCast(alpha_type, common_type), InvalidArgument, out);

  if (common_type == ScalarType::Half) {
    common_type = ScalarType::Float;
  }

  constexpr auto name = "sub.Scalar_out";

  ET_SWITCH_REALH_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_SCALAR_OBJ_REAL_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_IN = typename utils::promote_type_with_scalar_type<
          CTYPE_A,
          CTYPE_B,
          /*half_to_float*/ true>::type;
      ET_DCHECK(CppTypeToScalarType<CTYPE_IN>::value == common_type);

      CTYPE_B b_val;
      utils::extract_scalar(b, &b_val);
      CTYPE_IN b_casted = static_cast<CTYPE_IN>(b_val);

      CTYPE_IN alpha_val;
      utils::extract_scalar(alpha, &alpha_val);

      using CTYPE_OUT = typename std::conditional<
          std::is_same<CTYPE_A, internal::F2>::value,
          internal::F2,
          CTYPE_IN>::type;

      apply_unary_map_fn(
          [b_casted, alpha_val](const CTYPE_A val_a) {
            CTYPE_IN a_casted = static_cast<CTYPE_IN>(val_a);
            CTYPE_IN value = a_casted - alpha_val * b_casted;
            return static_cast<CTYPE_OUT>(value);
          },
          a.const_data_ptr<CTYPE_A>(),
          out.mutable_data_ptr<CTYPE_OUT>(),
          out.numel());
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
