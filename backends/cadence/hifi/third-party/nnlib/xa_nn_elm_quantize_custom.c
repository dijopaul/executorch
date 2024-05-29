/*******************************************************************************
* Copyright (c) 2018-2022 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
#include "common_fpu.h"
#include "xa_nnlib_common.h"
#include "xa_nn_basic_state.h"
#include "xa_nnlib_common_macros.h"
#include <math.h>


WORD32 xa_nn_elm_dequantize_asym8u_f32(FLOAT32 * __restrict__ p_out,
                                       const UWORD8 * __restrict__ p_inp,
                                       WORD32  inp_zero_bias,
                                       FLOAT32 inp_scale,
                                       WORD32  num_elm)
{
  int i;
  xtfloat *p_o = (xtfloat *)p_out;
  WORD8 *p_i = (WORD8 *)p_inp;

  ALIGN_REGISTER_TYPE align_inp;
  PRIME_8X4U(p_i, align_inp);

  ae_valign align_dst = AE_ZALIGN64();

  ae_int16x4 d_inp_zero_bias = AE_MOVDA16(inp_zero_bias);
  ae_int16x4 ONE = AE_MOVDA16(1);

  xtfloatx2 d_inp_scale = inp_scale;

  xtfloatx2 d_out0, d_out1;

  for(i = 0; i < (num_elm >> 2); i++)
  {
    ae_int16x4 d_inp;
    ae_int16x4 d_inp16;
    ae_int32x2 d_inp32_0, d_inp32_1;

    AE_LA8X4U_IP(d_inp, align_inp, p_i);

    d_inp16 =  AE_SUB16(d_inp, d_inp_zero_bias);

    AE_MUL16X4(d_inp32_0, d_inp32_1, d_inp16, ONE);

    d_out0 = XT_MUL_SX2(d_inp32_0, d_inp_scale);
    d_out1 = XT_MUL_SX2(d_inp32_1, d_inp_scale);

    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out0), align_dst, (ae_int32x2 *)p_o);
    AE_SA32X2_IP(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_out1), align_dst, (ae_int32x2 *)p_o);
  }
  AE_SA64POS_FP(align_dst, p_o);

  /*Remainder loop*/
  for(i = 0; i < (num_elm & 3); i++)
  {
    UWORD8 d_inp = *p_i;
    p_i++;

    //WORD16 d_inp16 = d_inp - (WORD16)inp_zero_bias;

    FLOAT32 d_float = d_inp * inp_scale;

    *p_o = d_float;
    p_o++;
  }
  return 0;
}



WORD32 xa_nn_elm_quantize_f32_asym8u(UWORD8 * __restrict__ p_out,
                                     const FLOAT32 * __restrict__ p_inp,
                                     FLOAT32 out_scale,
                                     WORD32  out_zero_bias,
                                     WORD32  num_elm)
{
  int i;
  xtfloatx2 *p_i = (xtfloatx2 *)p_inp;
  UWORD8 *p_o = (UWORD8 *)p_out;
  ae_int32x2 d_out_zero_bias = AE_MOVDA32(out_zero_bias);
  xtfloat *out_scale_ptr = &out_scale;
  ae_valign align_inp = AE_LA64_PP(p_inp);
  ae_int32x2 quant_max = AE_MOVDA32(255);
  ae_int32x2 quant_min = AE_MOVDA32(0);
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
  ALIGN_REGISTER_TYPE align_out = AE_ZALIGN64();
#endif
  for(i = 0; i < (num_elm >> 2); i++)
  {
    xtfloatx2 d_inp0, d_inp1;
    xtfloatx2 d_inp0_t, d_inp1_t;
    ae_int32x2 d_out32_0, d_out32_1;
    xtfloatx2 d_out_scale = (xtfloatx2)*out_scale_ptr;

    XT_LASX2IP(d_inp0, align_inp, p_i);
    XT_LASX2IP(d_inp1, align_inp, p_i);
    d_inp0_t = XT_DIV_SX2(d_inp0, d_out_scale);
    d_inp1_t = XT_DIV_SX2(d_inp1, d_out_scale);
    d_inp0_t = XT_FIROUND_SX2(d_inp0_t);
    d_inp1_t = XT_FIROUND_SX2(d_inp1_t);    
    d_out32_0 = XT_UTRUNC_SX2(d_inp0_t, 0);
    d_out32_1 = XT_UTRUNC_SX2(d_inp1_t, 0);
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    d_out32_1 = AE_ADD32S(d_out32_1, d_out_zero_bias);
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
    // clamped_out
    ae_int8x8 clamped = AE_SAT8X4X32_H(d_out32_0, d_out32_1);
    // Store Output
    AE_SAV8X8_XP(clamped, align_out, (ae_int8x8 *)p_o, 4);
#else
    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    CLAMP_VAL(d_out32_1, d_out32_1, quant_min, quant_max);
    STORE_8X4_FROM_32X4(p_o, d_out32_0, d_out32_1)
#endif
  }
#if (XCHAL_HAVE_HIFI1 &( XCHAL_HW_VERSION >= RI9_HWVERSION ))
    AE_SA64POS_FP(align_out, p_o);
#endif
  for(i = 0; i < (num_elm & 3) ; i++)
  {
    xtfloat d_out_scale = (xtfloat)*out_scale_ptr;
    xtfloat d_inp0;
    xtfloat d_inp0_t;
    ae_int32x2 d_out32_0;
    XT_LSIP(d_inp0, (xtfloat *)p_i, sizeof(FLOAT32));
    d_inp0_t = XT_DIV_S(d_inp0, d_out_scale);
    d_inp0_t = XT_FIROUND_S(d_inp0_t);
    d_out32_0 = XT_UTRUNC_S(d_inp0_t, 0);    
    d_out32_0 = AE_ADD32S(d_out32_0, d_out_zero_bias);
    CLAMP_VAL(d_out32_0, d_out32_0, quant_min, quant_max);
    *p_o = (WORD8)AE_MOVAD32_L(d_out32_0);
    p_o++;
  }
  
  return 0;

}

