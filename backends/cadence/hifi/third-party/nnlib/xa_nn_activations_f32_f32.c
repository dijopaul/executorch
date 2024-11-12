/*******************************************************************************
* Copyright (c) 2018-2024 Cadence Design Systems, Inc.
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
/* Common helper macros. */
#include "xa_nnlib_common_fpu.h"
#include "xa_type_def.h"
#include "../include/NatureDSP_Signal_math.h"
#include "xa_nn_common.h"
#include "xa_nnlib_common.h"
#include "xa_nnlib_err_chk.h"
#include <math.h>
#include <errno.h>

//#define LIMIT_SX2(out, inp, min, max){\
//        out = XT_MAX_SX2(min, inp);\
//        out = XT_MIN_SX2(max, out);\
//}

const union ufloat32uint32 ALIGN(8) xa_nnlib_pow2f_coef_1[] =
{
  { 0x39222a65 },
  { 0x3aaf931c },
  { 0x3c1d94fc },
  { 0x3d63578a },
  { 0x3e75fdf0 },
  { 0x3f317218 },
  { 0x3f800000 }

 //{ 0x3aaf931b },
 //{ 0x3c1e7220 },
 //{ 0x3d63578a },
 //{ 0x3e75fcc9 },
 //{ 0x3f317218 },
 //{ 0x3f800000 }

};

const union ufloat32uint32 ALIGN(8) xa_nnlib_polytanhf_tbl[]=
{
    {0x3c86a7d1UL},/* 1.6437442973e-002*/
    {0xbd57b3abUL},/*-5.2661579102e-002*/
    {0x3e086615UL},/* 1.3320191205e-001*/
    {0xbeaaaa0fUL} /*-3.3332869411e-001*/
};

const union ufloat32uint32 ALIGN(8) xa_nnlib_log2_e[2] =
{
  { 0x3fb8aa3b }, /* 1.4426950216      */
  { 0x32a57060 }  /* 1.9259629891e-008 */
};

const union ufloat32uint32 xa_nnlib_halfln3={0x3f0c9f54UL} ; /* log(3)/2 - tanh(log(3)/2)==0.5 */

#if !HAVE_VFPU && !HAVE_FPU
DISCARD_FUN_FOR_NONVOID_RETURN(float32_t,xa_nnlib_scl_tanhf,(float32_t x))
#else
float32_t xa_nnlib_scl_tanhf( float32_t x )
{
    float32_t zero, one, two, half, z, r, eps;
    float32_t y;
    float32_t p0, dy, y1;
    int32_t ux;
    int32_t e1, e2;
    int32_t SCF; /* Floating-point Status and Control Register values. */
#if HAVE_VFPU
    if ( xtbool2_extract_0(XT_UN_SX2(x,x)) )
#else
    if ( XT_UN_S(x,x))
#endif
    {
        __Pragma( "frequency_hint never" );
        errno = EDOM;
        return XT_ADD_S(x,x);
    }

    SCF = XT_RUR_FSR(); /* Sample floating-point exception flags. */

    zero = (float32_t)XT_CONST_S(0);
    one = (float32_t)XT_CONST_S(1);
    two = (float32_t)XT_CONST_S(2);
    half = (float32_t)XT_CONST_S(3);
    ux = XT_RFR(x);
    ux = (ux & 0x80000000);
    x = XT_ABS_S(x);
    if (x > xa_nnlib_halfln3.f)
    {
        /*
        * For a large input value tanh(x) is computed from exp(2*x)/2, using
        * the following identity: tanh(x) == 1 - 2/(exp(2*x)+1)
        */
        r = zero; XT_MADDN_S(r, two, x); x = r;
        {
            xtfloat t=(xtfloat)80.f;
            x = XT_MIN_S(x, t);
        }

        /* scale input to 1/ln(2) */
        p0 = XT_MUL_S(x, xa_nnlib_log2_e[0].f);
        #if defined(XT_FIROUND_S)
        p0 = XT_FIROUND_S(p0);
        #else
        p0 = XT_FLOAT_S(XT_ROUND_S(p0, 0), 0);
        #endif
        dy = XT_NEG_S(p0);
        XT_MADD_S(dy, x, xa_nnlib_log2_e[0].f);
        XT_MADD_S(dy, x, xa_nnlib_log2_e[1].f);
        /* compute 2^x */
        {
            float32_t y0, y2, y3, y4, y5, y6, dy2;
            dy2 = XT_MUL_S(dy, dy);
            y0 = xa_nnlib_pow2f_coef_1[0].f;
            y1 = xa_nnlib_pow2f_coef_1[1].f;
            y2 = xa_nnlib_pow2f_coef_1[2].f;
            y3 = xa_nnlib_pow2f_coef_1[3].f;
            y4 = xa_nnlib_pow2f_coef_1[4].f;
            y5 = xa_nnlib_pow2f_coef_1[5].f;
            y6 = xa_nnlib_pow2f_coef_1[6].f;
            XT_MADD_S(y1, y0, dy);
            XT_MADD_S(y3, y2, dy);
            XT_MADD_S(y5, y4, dy);

            XT_MADD_S(y3, y1, dy2);
            XT_MADD_S(y5, y3, dy2);
            XT_MADD_S(y6, y5, dy);
            y = y6;
        }

        /* resulted scaling */
        {
            xtfloat t;
            t=(xtfloat) 129.f;p0=XT_MIN_S(p0,t);
            t=(xtfloat)-151.f;p0=XT_MAX_S(p0,t);
        }

        /* Apply exponential part to the result */
        {
            uint32_t tmp, v1, v2;
            tmp = XT_TRUNC_S(p0, 0);
            tmp = tmp+254 - 30 - 1;
            v1 = (tmp>>1);
            v2 = (tmp-v1);
            e1 = (v1<<23);
            e2 = (v2<<23);
        }

        /*
        * Convert (y*2^(ex-30))/2 to floating-point p == exp(x)/2
        */
        r = XT_MUL_S(y, 1073741824.f);
        y = XT_MUL_S(r, XT_WFR(e2));
        y = XT_MUL_S(y, XT_WFR(e1));
        z = XT_ADD_S(y, half);
        /* Initial approximation for 1/y */
        r = XT_RECIP0_S(z);
        /* 2 Newton-Raphson iterations for 1/z  */
        eps = one; XT_MSUB_S(eps, z, r);
        XT_MADD_S(r, r, eps);
        eps = one; XT_MSUB_S(eps, z, r);
        XT_MADD_S(r, r, eps);
        z = XT_SUB_S(one, r);
    }
    else
    {
        /*
        * Use polynomial approximation for small input values. This branch is
        * also used for a NaN on input.
        */

        float32_t x2, x3, tn0, tn1, tn2, tn3;
        x2 = XT_MUL_S(x, x);
        x3 = XT_MUL_S(x, x2);
        tn0 = xa_nnlib_polytanhf_tbl[0].f;
        tn1 = xa_nnlib_polytanhf_tbl[1].f;
        tn2 = xa_nnlib_polytanhf_tbl[2].f;
        tn3 = xa_nnlib_polytanhf_tbl[3].f;
        XT_MADD_S(tn1, tn0, x2);
        XT_MADD_S(tn2, tn1, x2);
        XT_MADD_S(tn3, tn2, x2);
        z = x;
        XT_MADD_S(z, tn3, x3);
    }
    /* apply sign */
    XT_MOVT_S(z,XT_NEG_S(z),AE_MOVBA(((uint32_t)ux)>>31));

    XT_WUR_FSR(SCF);
    return (z);
} /* tanhf() */
#endif

void xa_nnlib_vec_gelu_approx_tanh(float32_t* restrict y, const float32_t* restrict x, int N)
{
#define SCR_SZ (MAX_ALLOCA_SZ/(2*sizeof(float32_t)))
    float32_t ALIGN(8) scratch[SCR_SZ];
    const ae_int32* restrict pPolytanhf=(const ae_int32*)xa_nnlib_polytanhf_tbl;
          xtfloatx2 * restrict pScrWr;
    const xtfloatx2 * restrict pScrRd;
    ae_valign aX,aY;
    const xtfloatx2* restrict pX;
          xtfloatx2* restrict pY;
    xtfloatx2 one = XT_CONST_S(1);
    xtfloatx2 two = XT_CONST_S(2);
    xtfloatx2 half = XT_CONST_S(3);
    
    xtfloatx2 k_Beta = (float)(M_SQRT2 * M_2_SQRTPI * 0.5);
    xtfloatx2 k_Kappa = (float)(0.044715);
    xtfloatx2 point_5 = 0.5f;

    int n, m, M;
    if(N <= 0) return;
    if(N&1)
    {
        *y++ = xa_nnlib_scl_tanhf(*x++); N--;
    }
    if(N <= 0) return;
    for(m = 0; m < N; m += SCR_SZ/2,x += SCR_SZ/2,y += SCR_SZ/2)
    {
        M = XT_MIN(N - m, SCR_SZ/2);
        /*
        * For a large input value tanh(x) is computed from exp(2*x)/2, using
        * the following identity: tanh(x) == 1 - 2/(exp(2*x)+1)
        */
        /* argumant reduction phase */
        pX    = (const xtfloatx2*)x;
        aX = AE_LA64_PP(pX);
        pScrWr = (xtfloatx2*)scratch;
        for(n = 0; n < (M>>1); n++)
        {
            xtfloatx2 d, p0, dy,t, cb;
            XT_LASX2IP(d, aX, pX);
            cb = XT_MUL_SX2(d, d);
            cb = XT_MUL_SX2(d, cb);
            cb = XT_MUL_SX2(k_Kappa, cb);
            cb = ADD_SX2(cb, d);
            cb = XT_MUL_SX2(cb, k_Beta);

            d = cb;

            d = XT_ABS_SX2(d);
            d = XT_MUL_SX2(two, d);
            t = (xtfloatx2)80.f; d = XT_MIN_SX2(d, t);

            /* scale input to 1/ln(2) */
            p0 = XT_MUL_SX2(d, xa_nnlib_log2_e[0].f);
            #if defined(XT_FIROUND_SX2)
            p0 = XT_FIROUND_SX2(p0);
            #else
            p0 = XT_FLOAT_SX2(XT_ROUND_SX2(p0, 0), 0);
            #endif
            dy = XT_NEG_SX2(p0);

            XT_MADD_SX2(dy, d, xa_nnlib_log2_e[0].f);
            XT_MADD_SX2(dy, d, xa_nnlib_log2_e[1].f);
            XT_SSX2IP(dy ,pScrWr,sizeof(xtfloatx2));
            /* saturating p0 to the right values */
            t = (xtfloatx2) 129.f; p0 = XT_MIN_SX2(p0,t);
            t = (xtfloatx2) - 151.f; p0 = XT_MAX_SX2(p0,t);
            XT_SSX2IP(p0,pScrWr,sizeof(xtfloatx2));
        }
        /* compute 2^x via polynomal appoximation */
        __Pragma("no_reorder")
        pScrRd = (const xtfloatx2*)scratch;
        pScrWr = (xtfloatx2*)scratch;
        pPolytanhf = (const ae_int32*)xa_nnlib_pow2f_coef_1;
        for(n = 0; n < (M>>1); n++)
        {
            xtfloatx2 dy, y0,y1, y2, y3, y4, y5, y6, y7, dy2;
            ae_int32x2 tmp;
            XT_LSX2IP(dy ,pScrRd,2*sizeof(xtfloatx2));
            dy2 = XT_MUL_SX2(dy, dy);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           y3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-4*(int)sizeof(float32_t));   y4 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            y5 = xa_nnlib_pow2f_coef_1[5].f;
            y6 = xa_nnlib_pow2f_coef_1[6].f;
            XT_MADD_SX2(y1, y0, dy);
            XT_MADD_SX2(y3, y2, dy);
            XT_MADD_SX2(y5, y4, dy);
            XT_MADD_SX2(y3, y1, dy2);
            XT_MADD_SX2(y5, y3, dy2);
            XT_MADD_SX2(y6, y5, dy);
            y7 = y6;
            XT_SSX2IP(y7 ,pScrWr,2*sizeof(xtfloatx2));
        }
        /* resulted scaling by 2^N and final Newton-Raphson phase */
        __Pragma("no_reorder")
        pScrRd = (const xtfloatx2*)scratch;
        pScrWr = (xtfloatx2*)scratch;
        for(n = 0; n < (M>>1); n++)
        {
            xtfloatx2  d, z, r, eps, p0;
            ae_int32x2 tmp, v1, v2, e1, e2;
            XT_LSX2IP(d ,pScrRd,sizeof(xtfloatx2));
            XT_LSX2IP(p0,pScrRd,sizeof(xtfloatx2));

            /* Apply exponential part to the result */
            tmp = XT_TRUNC_SX2(p0, 0);
            tmp = AE_ADD32(tmp,254 - 1);
            v1 = AE_SRLI32(tmp,1);
            v2 = AE_SUB32(tmp,v1);
            e1 = AE_SLLI32(v1,23);
            e2 = AE_SLLI32(v2,23);
            /*
            * Convert (y*2^(ex-30))/2 to floating-point p == exp(x)/2
            */
            d = XT_MUL_SX2(d, XT_AE_MOVXTFLOATX2_FROMINT32X2(e2));
            d = XT_MUL_SX2(d, XT_AE_MOVXTFLOATX2_FROMINT32X2(e1));
            z = XT_ADD_SX2(d, half);
            /* Initial approximation for 1/y */
            r = XT_RECIP0_SX2(z);
            /* 2 Newton-Raphson iterations for 1/z  */
            eps = one; XT_MSUB_SX2(eps, z, r);
            XT_MADD_SX2(r, r, eps);
            eps = one; XT_MSUB_SX2(eps, z, r);
            XT_MADD_SX2(r, r, eps);
            z = XT_SUB_SX2(one, r);
            XT_SSX2IP(z,pScrWr,2*sizeof(xtfloatx2));
        }
        /* next, compute output for smaller argument
           Use polynomial approximation for small input values. This branch is
           also used for a NaN on input.
        */
        __Pragma("no_reorder")
        pX    = (const xtfloatx2*)x;
        pScrWr = (( xtfloatx2*)scratch)+1;
        aX = AE_LA64_PP(pX);
        pPolytanhf = (const ae_int32*)xa_nnlib_polytanhf_tbl;
        for(n = 0; n < (M>>1); n++)
        {
            xtfloatx2 z, x1, x2, x3, tn0, tn1, tn2, tn3, cb;
            XT_LASX2IP(x1,aX,pX);
            cb = XT_MUL_SX2(x1, x1);
            cb = XT_MUL_SX2(x1, cb);
            cb = XT_MUL_SX2(k_Kappa, cb);
            cb = ADD_SX2(cb, x1);
            cb = XT_MUL_SX2(cb, k_Beta);
            x1 = cb;
            x1 = XT_ABS_SX2(x1);
            x2 = XT_MUL_SX2(x1, x1);
            x3 = XT_MUL_SX2(x1, x2);
            ae_int32x2 tmp;
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn0 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn1 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_IP(tmp,pPolytanhf,sizeof(float32_t));           tn2 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            AE_L32_XP(tmp,pPolytanhf,-3*(int)sizeof(float32_t));   tn3 = XT_AE_MOVXTFLOATX2_FROMINT32X2(tmp);
            XT_MADD_SX2(tn1, tn0, x2);
            XT_MADD_SX2(tn2, tn1, x2);
            XT_MADD_SX2(tn3, tn2, x2);
            z = x1;
            XT_MADD_SX2(z, tn3, x3);
            XT_SSX2IP(z,pScrWr,2*sizeof(xtfloatx2));
        }
        /* final stage: select right output and apply sign */
        __Pragma("no_reorder")
        pX    = (const xtfloatx2*)x;
        pY    = (xtfloatx2*)y;
        pScrRd = (const xtfloatx2*)scratch;
        aX = AE_LA64_PP(pX); aY = AE_ZALIGN64();
        for(n = 0; n < (M>>1); n++)
        {
            xtbool2 bbig,bsign;
            xtfloatx2 d, z, zbig, cb, inp;
            ae_int32x2 ux;
            XT_LASX2IP(d,aX,pX);
            inp = d;
            cb = XT_MUL_SX2(d, d);
            cb = XT_MUL_SX2(d, cb);
            cb = XT_MUL_SX2(k_Kappa, cb);
            cb = ADD_SX2(cb, d);
            cb = XT_MUL_SX2(cb, k_Beta);
            d = cb;
            ux = XT_AE_MOVINT32X2_FROMXTFLOATX2(d);
            bsign = AE_LT32(ux,0);
            d = XT_ABS_SX2(d);
            bbig = XT_OLT_SX2(xa_nnlib_halfln3.f,d);
            XT_LSX2IP(zbig,pScrRd,sizeof(xtfloatx2));
            XT_LSX2IP(z   ,pScrRd,sizeof(xtfloatx2));
            XT_MOVT_SX2(z,zbig,bbig);
            /* apply sign */
            XT_MOVT_SX2(z,XT_NEG_SX2(z),bsign);
            z = ADD_SX2(one, z);
            z = XT_MUL_SX2(z, point_5);
            z = XT_MUL_SX2(z, inp);
            XT_SASX2IP(z,aY,pY);
        }
        AE_SA64POS_FP(aY,pY);
    }
}

const union ufloat32uint32 ALIGN(8) __erff_tbl[]=
{
        {0x40754fdf}, /* 3.833 - limit  */
        {0x3f8ccccd}, /* 1.1 */
        {0x3df75cae}, /* log(2/sqrt(pi)) */
        /* polynomial for small x:
            x=(-1.15:pow2(1,-13):1.15);
            order=5;

            x(find(x==0))=[];
            y=erf(x);
            x2=x.^2;
            z=f_fma_emu(0.25,x2,log(y*(sqrt(pi)/2)./x));
            p=polyfit(x2,z,order);
                p(end)=0;
            for k=1:7
                d=z-polyval(p,x2);
                dp=polyfit(x2,d,order);
                p=p+dp;
                p(end)=0;
            end
            p(end-1)=p(end-1)-0*eps(p(end-1));
        */
        {0x38753d56},
        {0xb98a19bc},
        {0xbb39d314},
        {0x3d360fc5},
        {0xbdaaaae8},

        /* polynomial for big x:
            xlow=1.1;
            x=(xlow:1e-5:1.9);
            order=5;
            r=1./x;
            t=log(x.*erfc(x))+x.^2;
            p=polyfit(r,t,order);
            for k=1:3
                d=t-polyval(p,r);
                dp=polyfit(r,d,order);
                p=p+dp;
            end
        */
        {0x3d5b5d40},
        {0xbe8e4957},
        {0x3f1e1c5b},
        {0xbf33984c},
        {0x3d07dd45},
        {0xbf131274},

        /* polynomial and constants for exp() */
        {0x3fb8aa3b},
        {0x32a57060},
        {0xb9222a6c},
        {0xbaaf9326},
        {0xbc1d94fc},
        {0xbd63578a},
        {0xbe75fdf0},
        {0xbf317218},
};


void xa_nnlib_vec_gelu_no_approx(float32_t* restrict y, const float32_t* restrict x, int N)
{
       ae_valign aX,aY;
       xtfloatx2 one = XT_CONST_S(1);
       xtfloatx2 point_5 = 0.5f;
       const xtfloatx2* restrict pX;
       xtfloatx2* restrict pY;
       xtfloatx2 M_SQRT_1_2 = (float)(M_SQRT1_2);
        __Pragma("no_reorder")
        pX    = (const xtfloatx2*)x;
        pY    = (xtfloatx2*)y;
        aX=AE_LA64_PP(pX); aY=AE_ZALIGN64();
        for(int n = 0; n < (N>>1); n++)
        {
            xtfloatx2 d, d_seed, d_err, z;
            XT_LASX2IP(d,aX,pX);
            d_seed = XT_MUL_SX2(d, M_SQRT_1_2);
            xtbool2 bsmall;
            xtfloatx2 x2, y, zbig, r, zsmall;
            const union ufloat32uint32 * restrict erff_tbl_Q = __erff_tbl+8;

            int sx_h = AE_MOVAD32_H(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_seed));
            int sx_l = AE_MOVAD32_L(XT_AE_MOVINT32X2_FROMXTFLOATX2(d_seed));
            d_seed = XT_ABS_SX2(d_seed);
            bsmall = XT_OLT_SX2(d_seed, __erff_tbl[1].f);
            d_seed = XT_MIN_SX2(__erff_tbl[0].f, d_seed);
        #if __has_builtin(RECIPQLI_S)
            r = RECIPQLI_SX2(d_seed);
        #else
            r = XT_RECIP_SX2(d_seed);
        #endif
            x2 = XT_MUL_SX2(d_seed,d_seed);
            /* compute polynomial for small x */
            {
                xtfloatx2 x4,z0,z1,t;
                x4 = XT_MUL_SX2(x2,x2);
                z0 = __erff_tbl[3+0].f;
                z1 = __erff_tbl[3+1].f;
                t = __erff_tbl[3+2].f; XT_MADD_SX2(t,x4,z0);z0 = t;
                t = __erff_tbl[3+3].f; XT_MADD_SX2(t,x4,z1);z1 = t;
                t = __erff_tbl[3+4].f; XT_MADD_SX2(t,x4,z0);z0 = t;
                zsmall = z0; XT_MADD_SX2(zsmall, x2,z1);
                t = __erff_tbl[2].f; XT_MADD_SX2(t,x2,zsmall);zsmall = t;
                XT_MSUB_SX2(zsmall,x2,0.25f);
            }
            /* compute polynomial for big x */
            {   // this variant works as well
                xtfloatx2 r2,y0,y1,t;
                r2 = XT_MUL_SX2(r,r);
                y0 = erff_tbl_Q[0].f;
                y1 = erff_tbl_Q[1].f;
                t = erff_tbl_Q[2].f;XT_MADD_SX2(t,r2,y0);y0=t;
                t = erff_tbl_Q[3].f;XT_MADD_SX2(t,r2,y1);y1=t;
                t = erff_tbl_Q[4].f;XT_MADD_SX2(t,r2,y0);y0=t;
                t = erff_tbl_Q[5].f;XT_MADD_SX2(t,r2,y1);y1=t;
                y = y1; XT_MADD_SX2(y,r,y0);
                zbig = XT_SUB_SX2(y,x2);
            }
            y = zbig;
            XT_MOVT_SX2(y,zsmall,bsmall);
            /* compute exponent */
            {
                xtfloatx2 e,d,z0,z1,y2,t,s;
                ae_int32x2 n;
                e = XT_FIROUND_SX2(XT_MUL_SX2(y,__erff_tbl[14+0].f));
                d = XT_NEG_SX2(e); XT_MADD_SX2(d,y,__erff_tbl[14+0].f);
                XT_MADD_SX2(d,y,__erff_tbl[14+1].f);
                z0 = __erff_tbl[14+2].f;
                z1 = __erff_tbl[14+3].f;
                y2 = XT_MUL_SX2(d,d);
                t = __erff_tbl[14+4].f;XT_MADD_SX2(t,y2,z0);z0=t;
                t = __erff_tbl[14+5].f;XT_MADD_SX2(t,y2,z1);z1=t;
                t = __erff_tbl[14+6].f;XT_MADD_SX2(t,y2,z0);z0=t;
                t = __erff_tbl[14+7].f;XT_MADD_SX2(t,y2,z1);z1=t;
                XT_MADD_SX2(z1,d,z0);
        #if __has_builtin(MULSONE_S)
                y = MULSONE_S(d,z1);
        #else
                y = XT_CONST_S(1); //1.f
                XT_MSUB_SX2(y, d, z1);
        #endif
                /* simplified ldexp(y,(int)e) */
                n = XT_TRUNC_SX2(e, 0);
                n = AE_ADD32(n, AE_MOVDA32X2(127, 127));
                n = n << 23;
                s = AE_MOVXTFLOATX2_FROMINT32X2 (n);
                y = XT_MUL_SX2(y,s);
            }
        #if __has_builtin(MULSONE_S)
            z = MULSONE_S(r,y);
        #else
            z = XT_CONST_S(1); //1.f
            XT_MSUB_SX2(z, r, y);
        #endif
            XT_MOVT_SX2(z,XT_MUL_SX2(y,d_seed),bsmall);
            xtfloatx2 neg_z = XT_NEG_SX2(z);
            xtfloat z_h, z_l;
            z_h = XT_HIGH_S(z);
            z_l = XT_LOW_S(z);
            XT_MOVLTZ_S(z_h, XT_HIGH_S(neg_z), sx_h);
            XT_MOVLTZ_S(z_l, XT_LOW_S(neg_z), sx_l);
            z = XT_SEL32_HL_SX2(z_h, z_l);
            d_err = ADD_SX2(one, z);
            d_err = XT_MUL_SX2(point_5, d_err);
            z = XT_MUL_SX2(d, d_err);
            XT_SASX2IP(z,aY,pY);
        }
        AE_SA64POS_FP(aY,pY);
}

WORD32 xa_nn_vec_gelu_f32_f32(
    FLOAT32       * __restrict__ p_out,        /* result, floating point */
    FLOAT32       * __restrict__ p_vec,        /* input data, floating point */
    WORD32        vec_length,                  /* length of vectors */
    WORD32        approx )                     /* approximation flag 0: none 1: tanh */

{

    if(approx == 1)
    {
        
        /* approximation is tanh */
        xa_nnlib_vec_gelu_approx_tanh(p_out, p_vec, vec_length);

    }
    else
    {
        /* approximation is none */
        xa_nnlib_vec_gelu_no_approx(p_out, p_vec, vec_length);
    }

    return 0;
}
