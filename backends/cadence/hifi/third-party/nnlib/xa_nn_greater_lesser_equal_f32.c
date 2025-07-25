#include "xa_type_def.h"
#include "xa_nnlib_common_fpu.h"
#include "xa_nn_common.h"
#include "xa_nnlib_err_chk.h"
//#include "xa_nn_basic_state.h"
#include "xa_nnlib_kernels_api.h"


#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_greater_lesser_equal_f32xf32_f32,
             (
                WORD8 *y,
                const FLOAT32 *x1,
                const FLOAT32 *x2,
                WORD32 N,
                WORD32 kernel_type
              )
           )
#else
WORD32 xa_nn_elm_greater_lesser_equal_f32xf32_f32(WORD8 * __restrict__ p_out,
                               const FLOAT32 * __restrict__ p_inp1,
                               const FLOAT32 * __restrict__ p_inp2,
                               WORD32 num_elm,
                               WORD32 kernel_type)
{
    /* NULL pointer checks */
    XA_NNLIB_ARG_CHK_PTR(p_out, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
    XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
    /* Pointer alignment checks */
    XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
    XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
    /* Basic Parameter checks */
    XA_NNLIB_ARG_CHK_COND((num_elm <= 0), -1);

    int i;
    xtfloatx2 *inp1 = (xtfloatx2 *)p_inp1;
    xtfloatx2 *inp2 = (xtfloatx2 *)p_inp2;
    //xtfloatx2 *out =  (xtfloatx2 *)p_out;
    UWORD8 *out = p_out;
    xtfloatx2 x1, x2, y;
    xtbool check;
    
    xtfloatx2 float_0 = XT_MOV_SX2(AE_ZERO32());

    if(kernel_type == 0)
    {    
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              /*y = XT_SUB_SX2(x2, x1);*/
              xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == 1)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              //y = XT_SUB_SX2(x2, x1);
              xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }  
    }
    else if(kernel_type == 2)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              //y = XT_SUB_SX2(x1, x2);
              xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a1, a2);
          
          check = 0;        
          if(a <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == 3)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              //y = XT_SUB_SX2(x1, x2);
              xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a1, a2);
          
          check = 0;        
          if(a < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == 4)
    {
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              //y = XT_SUB_SX2(x2, x1);
              xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
              
              uint8_t val = AE_MOVAB2(check);
              
              uint8_t store1 = (val >> 1) & 0x1;
              *out++ = store1;
              
              uint8_t store0 = val & 0x1;
              *out++ = store0;
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *out++ = store1;
            
            uint8_t store0 = val & 0x1;
            *out++ = store0;
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          //a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a1 == a2)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }
    else if(kernel_type == 5)
    {
      ae_int32x2 ones = AE_MOVDA32(1);
      if(((((unsigned)p_out)&7) == 0) && ((((unsigned)p_inp1)&7) == 0) && ((((unsigned)p_inp2)&7) == 0))
      {
          for(i=0;i < num_elm>>1;i++)
          {
              XT_LSX2IP(x1, inp1, 2*sizeof(FLOAT32));
              XT_LSX2IP(x2, inp2, 2*sizeof(FLOAT32));
              
              //y = XT_SUB_SX2(x2, x1);
              xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
              
              ae_int32x2 store = AE_ZERO32();
              AE_MOVF32X2(store, ones, check);
              
              *out++ = AE_MOVAD32_H(store);
              *out++ = AE_MOVAD32_L(store);
          }
      }
      else
      {
          ae_valign inp1_a, inp2_a, out_a;
  
          inp1_a = XT_LASX2PP(inp1);
          inp2_a = XT_LASX2PP(inp2);
          /* Each iteration of loop is independent so safe to use concurrent pragma */
#pragma concurrent
          for(i=0;i < num_elm>>1;i++)
          {
            XT_LASX2IP(x1, inp1_a, inp1);
            XT_LASX2IP(x2, inp2_a, inp2);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *out++ = AE_MOVAD32_H(store);
            *out++ = AE_MOVAD32_L(store);
          }
      }
      // Remainder Loop
      if (num_elm & 1)
      {
          xtfloat a1, a2, a;
          XT_LSIP(a1, (xtfloat *)inp1, 0);
          XT_LSIP(a2, (xtfloat *)inp2, 0);
          
          a = XT_SUB_S(a2, a1);
          
          check = 0;        
          if(a != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *out++ = store;
      }
    }

    return 0;
}
#endif

#if HAVE_VFPU
static void internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32(UWORD8 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  out_lc,
                             WORD32  in_lc,
                             xtbool  sign_flag,
                             WORD32 kernel_type)
{
  int i, j;

  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2; 
  
  xtbool check;
  
  xtfloatx2 float_0 = XT_MOV_SX2(AE_ZERO32());

  int num_simd2_ops;
  int num_scalar_ops;

  if(out_lc)
  {
    num_simd2_ops = in_lc >> 1;
    num_scalar_ops = in_lc & 1;
  }
  else
  {
    num_simd2_ops = (in_lc >> 2) << 1;
    num_scalar_ops = in_lc & 3;
  }

    xtfloatx2 x1, x2, y;
    xtfloat a0, b0, c0;

  /* For computing inp2 - inp1 */   
  if(sign_flag){  
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      UWORD8 *p_c = (UWORD8 *)&p_out[i * in_lc];
      
      if(kernel_type == 0)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 1)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 2)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 3)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 4)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          //c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(a0 == b0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 5)
      {
        ae_int32x2 ones = AE_MOVDA32(1);
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
    }
  }
  /* For computing inp1 - inp2 */   
  else
  {
    for(i = 0; i < out_lc; i++)
    {
      p_a = (xtfloatx2 *)&p_inp1[i * in_lc];
      p_b = (xtfloatx2 *)p_inp2;
      UWORD8 *p_c = (UWORD8 *)&p_out[i * in_lc];
      
      if(kernel_type == 0)
      {    
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if (kernel_type == 1)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 2)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 <= 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 3)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x1, x2);
            xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(a0, b0);   
          
          check = 0;
          
          if(c0 < 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 4)
      {
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            uint8_t val = AE_MOVAB2(check);
            
            uint8_t store1 = (val >> 1) & 0x1;
            *p_c++ = store1;
            
            uint8_t store0 = val & 0x1;
            *p_c++ = store0;
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          //c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(a0 == b0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
      else if(kernel_type == 5)
      {
        ae_int32x2 ones = AE_MOVDA32(1);
        if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_b)&7) == 0) && ((((unsigned)p_c)&7) == 0))
        {
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LSX2IP(x1, p_a, 2*sizeof(FLOAT32));
            XT_LSX2IP(x2, p_b, 2*sizeof(FLOAT32));
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        else
        {
          ae_valign vinp1, vinp2;
          vinp1 = XT_LASX2PP(p_a);
          vinp2 = XT_LASX2PP(p_b);
  
          for(j = 0; j < num_simd2_ops; j++)
          {
            XT_LASX2IP(x1, vinp1, p_a);
            XT_LASX2IP(x2, vinp2, p_b);
            
            //y = XT_SUB_SX2(x2, x1);
            xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
            
            ae_int32x2 store = AE_ZERO32();
            AE_MOVF32X2(store, ones, check);
            
            *p_c++ = AE_MOVAD32_H(store);
            *p_c++ = AE_MOVAD32_L(store);
          }
        }
        if(num_scalar_ops !=0)
        {
          XT_LSIP(a0, (xtfloat *)p_a, sizeof(FLOAT32));
          XT_LSIP(b0, (xtfloat *)p_b, sizeof(FLOAT32));
          c0 = XT_SUB_S(b0, a0);   
          
          check = 0;
          
          if(c0 != 0)
            check = 1;
          
          uint8_t store = AE_MOVAB(check);
          *p_c++ = store;
        }
      }
    }  
  }
}

static void internal_elm_greater_lesser_equal_broadcast_f32xf32_f32(UWORD8 * __restrict__ p_out,
                    const    FLOAT32 * __restrict__ p_inp1,
                    const    FLOAT32 * __restrict__ p_inp2,
                             WORD32  num_elm,
                             xtbool  sign_flag,
                             WORD32 kernel_type)
{
  int i;
  xtfloatx2  * __restrict__ p_a = (xtfloatx2 *)p_inp1;
  xtfloatx2  * __restrict__ p_b = (xtfloatx2 *)p_inp2; 
  
  xtbool check;
  
  UWORD8 * p_c = p_out;
  xtfloatx2 float_0 = XT_MOV_SX2(AE_ZERO32());

  const int num_simd2_ops = num_elm >> 1;
  const int num_scalar_ops = num_elm & 1;

  xtfloat a0_7, out;
  xtfloatx2 x1, x2, y;
  x2 = XT_LSI((xtfloat *)p_b, 0);
        
  /* For computing inp2 - inp1 */      
  if(sign_flag){
    if(kernel_type == 0)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 1)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 2)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 3)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 4)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out == 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 5)
    {
      ae_int32x2 ones = AE_MOVDA32(1);
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store); 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);   
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          ae_int32x2 store = AE_ZERO32();
          AE_MOVF32X2(store, ones, check);
          
          *p_c++ = AE_MOVAD32_H(store);
          *p_c++ = AE_MOVAD32_L(store);
        }  
      }  
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out != 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
  }
  /* For computing inp1 - inp2 */   
  else
  {
    if(kernel_type == 0)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 1)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x2, x1);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 2)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LE_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out <= 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 3)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x1, x2);
          
          xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x1, x2);
          
        xtbool2 check = xtfloatx2_LT_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(a0_7, x2);   
        
        check = 0;
          
        if(out < 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
    else if(kernel_type == 4)
    {
      if(((((unsigned)p_a)&7) == 0) && ((((unsigned)p_c)&7) == 0))
      {
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LSX2IP(x1, p_a, 2 * sizeof(FLOAT32));
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0; 
        }
      }
      else
      {
        ae_valign inp1_a, out_a;
        inp1_a = XT_LASX2PP(p_a);
        
        for(i=0; i<num_simd2_ops; i++)
        {
          XT_LASX2IP(x1, inp1_a, p_a);
          //y = XT_SUB_SX2(x2, x1);
          
          xtbool2 check = xtfloatx2_EQ_xtfloatx2(x1, x2);
          
          uint8_t val = AE_MOVAB2(check);
          
          uint8_t store1 = (val >> 1) & 0x1;
          *p_c++ = store1;
          
          uint8_t store0 = val & 0x1;
          *p_c++ = store0;
        }
      }
      if(num_scalar_ops !=0)
      {
        XT_LSIP(a0_7, (xtfloat *)p_a, sizeof(FLOAT32));
        out = XT_SUB_S(x2, a0_7);   
        
        check = 0;
          
        if(out == 0)
          check = 1;
          
        uint8_t store = AE_MOVAB(check);
        *p_c++ = store;
      }
    }
  }
}
#endif

#if !HAVE_VFPU
DISCARD_FUN_FOR_NONVOID_RETURN(
             WORD32, xa_nn_elm_greaterequal_broadcast_4D_f32xf32_f32,
             (
                      WORD8 * p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * p_inp2,
                      const WORD32 *const p_inp2_shape,
                      WORD32 kernel_type
              )
           )
#else           
WORD32 xa_nn_elm_greater_lesser_equal_broadcast_4D_f32xf32_f32(WORD8 * __restrict__ p_out,
                      const WORD32 *const p_out_shape,
                      const FLOAT32 * __restrict__ p_inp1,
                      const WORD32 *const p_inp1_shape,
                      const FLOAT32 * __restrict__ p_inp2,
                      const WORD32 *const p_inp2_shape,
                      WORD32 kernel_type)
{
  /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2, -1);
  XA_NNLIB_ARG_CHK_PTR(p_out_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp1_shape, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp2_shape, -1);
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2, sizeof(FLOAT32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_out_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp1_shape, sizeof(WORD32), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp2_shape, sizeof(WORD32), -1);

  /* Check shapes */
  int i;
  xtbool sign_flag;
  for(i = 0; i < 4; i++)
  {
    if((p_inp1_shape[i] != p_inp2_shape[i] && p_inp1_shape[i] != 1 && p_inp2_shape[i] != 1) ||
       (p_out_shape[i] != (p_inp1_shape[i] > p_inp2_shape[i] ? p_inp1_shape[i] : p_inp2_shape[i])))
    {
      return -1;
    }
  }

  WORD32 inp1_strides[4], inp2_strides[4];
  inp1_strides[3] = 1;
  inp2_strides[3] = 1;
  for(i = 2; i >= 0; i--)
  {
    ae_int32x2 d_str, d_shape;
    d_str = AE_MOVDA32X2(inp1_strides[i + 1], inp2_strides[i + 1]);
    d_shape = AE_MOVDA32X2(p_inp1_shape[i + 1], p_inp2_shape[i + 1]);
    d_str = AE_MULP32X2(d_str, d_shape);
    inp1_strides[i] = AE_MOVAD32_H(d_str);
    inp2_strides[i] = AE_MOVAD32_L(d_str);
  }

  int need_broadcast = 0;
  int inp1_const = 1, inp2_const = 1;
  for(i = 0; i < 4; i++)
  {
    if(p_inp1_shape[i] != p_inp2_shape[i])
    {
      if(p_inp1_shape[i] == 1)
        inp1_strides[i] = 0;
      else
        inp2_strides[i] = 0;

      need_broadcast = 1;
    }
    if(p_inp1_shape[i] != 1)
      inp1_const &= 0;
    if(p_inp2_shape[i] != 1)
      inp2_const &= 0;
  }
  int itr0, itr1, itr2;

  UWORD8 *p_out_tmp = p_out;
  const FLOAT32 *__restrict__ p_inp1_tmp = p_inp1;
  const FLOAT32 *__restrict__ p_inp2_tmp = p_inp2;
  if(need_broadcast == 0)
  {
    sign_flag = 0;
    internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32(
                p_out,
                p_inp1,
                p_inp2,
                1,
                p_out_shape[0] * inp1_strides[0],
                sign_flag,
                kernel_type);
  }
  else if(inp1_strides[3] == inp2_strides[3])
  {
    WORD32 in_lc, out_lc;
    sign_flag = 0;
    in_lc = p_out_shape[2] * p_out_shape[3];
    out_lc = 1;
    if(inp1_strides[2] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
      sign_flag = 1;
      int tmp_strides[2];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }
    else if(inp2_strides[2] == 0)
    {
      in_lc = p_out_shape[3];
      out_lc = p_out_shape[2];
    }

    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const FLOAT32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const FLOAT32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        internal_elm_greater_lesser_equal_broadcast_2D_f32xf32_f32(
            p_out_tmp,
            p_inp1_tmp0,
            p_inp2_tmp0,
            out_lc,
            in_lc,
            sign_flag,
            kernel_type);
        p_out_tmp += in_lc * out_lc;
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  else if(inp1_const == 1 || inp2_const == 1)
  {
    sign_flag = 0;
    if(inp1_strides[3] == 0)
    {
      sign_flag = 1;
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
    }
    internal_elm_greater_lesser_equal_broadcast_f32xf32_f32(
        p_out_tmp,
        p_inp1_tmp,
        p_inp2_tmp,
        p_out_shape[0] * p_out_shape[1] * p_out_shape[2] * p_out_shape[3],
        sign_flag,
        kernel_type);
  }
  else
  {
    sign_flag = 0;
    if(inp1_strides[3] == 0)
    {
      const FLOAT32 *tmp;
      tmp = p_inp1_tmp;   p_inp1_tmp = p_inp2_tmp;    p_inp2_tmp = tmp;
      sign_flag = 1;
      int tmp_strides[3];
      tmp_strides[0] = inp1_strides[0];
      tmp_strides[1] = inp1_strides[1];
      tmp_strides[2] = inp1_strides[2];

      inp1_strides[0] = inp2_strides[0];
      inp1_strides[1] = inp2_strides[1];
      inp1_strides[2] = inp2_strides[2];

      inp2_strides[0] = tmp_strides[0];
      inp2_strides[1] = tmp_strides[1];
      inp2_strides[2] = tmp_strides[2];
    }
    for(itr0 = 0; itr0 < p_out_shape[0]; itr0++)
    {
      const FLOAT32 *__restrict__ p_inp1_tmp0 = p_inp1_tmp;
      const FLOAT32 *__restrict__ p_inp2_tmp0 = p_inp2_tmp;
      for(itr1 = 0; itr1 < p_out_shape[1]; itr1++)
      {
        const FLOAT32 *__restrict__ p_inp1_tmp1 = p_inp1_tmp0;
        const FLOAT32 *__restrict__ p_inp2_tmp1 = p_inp2_tmp0;
        for(itr2 = 0; itr2 < p_out_shape[2]; itr2++)
        {
          {
            internal_elm_greater_lesser_equal_broadcast_f32xf32_f32(
                p_out_tmp,
                p_inp1_tmp1,
                p_inp2_tmp1,
                p_out_shape[3], 
                sign_flag,
                kernel_type);
          }
          p_out_tmp += p_out_shape[3];
          p_inp1_tmp1 += inp1_strides[2];
          p_inp2_tmp1 += inp2_strides[2];
        }
        p_inp1_tmp0 += inp1_strides[1];
        p_inp2_tmp0 += inp2_strides[1];
      }
      p_inp1_tmp += inp1_strides[0];
      p_inp2_tmp += inp2_strides[0];
    }
  }
  return 0;
}
#endif
