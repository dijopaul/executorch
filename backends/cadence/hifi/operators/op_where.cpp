/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& where_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ScalarType cond_type = cond.scalar_type();
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "where.self_out";
  
  //printf("a num = %d\n", a.numel());
  //printf("b num  = %d\n", b.numel());
  //printf("out num = %d\n", out.numel());
  
    //printf("a dim = %zu\n", a.dim());
    //printf("b dim = %zu\n", b.dim());
    //printf("cond dim = %zu\n", cond.dim());
    //printf("out dim = %zu\n", out.dim());
    
  /*int i;
  for(i = 0; i < out.dim(); i++)
      printf(" out.size(%d):%d ", i, out.size(i));
  for(i = 0; i < a.dim(); i++)
      printf(" a.size(%d):%d ", i, a.size(i));
  for(i = 0; i < b.dim(); i++)
      printf(" b.size(%d):%d ", i, b.size(i));
  for(i = 0; i < cond.dim(); i++)
      printf(" cond.size(%d):%d ", i, cond.size(i));
  printf("\n");*/

  ET_CHECK_MSG(
      cond_type == ScalarType::Bool || cond_type == ScalarType::Byte,
      "Unhandled dtype %s for where.self_out",
      torch::executor::toString(cond_type));
  ET_SWITCH_REALHB_TYPES(a_type, ctx, name, CTYPE_A, [&]() {
    ET_SWITCH_REALHB_TYPES(b_type, ctx, name, CTYPE_B, [&]() {
      using CTYPE_OUT =
          typename torch::executor::promote_types<CTYPE_A, CTYPE_B>::type;
      apply_ternary_elementwise_fn<CTYPE_A, CTYPE_B, uint8_t, CTYPE_OUT>(
          [](const CTYPE_A val_a, const CTYPE_B val_b, const uint8_t val_c) {
            CTYPE_OUT a_casted = static_cast<CTYPE_OUT>(val_a);
            CTYPE_OUT b_casted = static_cast<CTYPE_OUT>(val_b);
            return val_c ? a_casted : b_casted;
          },
          a,
          b,
          cond,
          out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
