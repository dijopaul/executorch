#include "xa_nnlib_common.h"
#include "xa_nnlib_common_macros.h"
#include "xa_nn_conv2d_std_state.h"

static WORD32 conv_x_left_pad(
    WORD32 x_padding,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 out_width_over_x_pad = (x_padding - kernel_width)/x_stride + 1;
  WORD32 left_shift, right_shift;
  out_width_over_x_pad = out_width_over_x_pad > out_width ? out_width : out_width_over_x_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(255);
  ae_int32x2 min_int8 = AE_MOVDA32(0);

  /* When kernel convolves over x-left pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = 0; j < out_width_over_x_pad; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */          
        ae_int32x2 acc;
#if XCHAL_HAVE_HIFI1
        if(p_bias != NULL){
          acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else
        if(p_bias != NULL){
          acc = AE_MOVDA32(p_bias[k]);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
      }
    }
  }
  return out_width_over_x_pad;
}

static WORD32 conv_x_right_pad(
    WORD32 x_padding,
    WORD32 input_width,
    WORD32 x_stride,
    WORD32 out_width,
    WORD32 out_height,
    WORD32 out_channels,
    WORD32 out_channels_offset,
    WORD32 out_width_offset,
    WORD32 out_height_offset,
    const WORD32* __restrict__ p_bias,
    WORD8 *p_out,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias)
{
  WORD32 i,j,k;
  WORD32 idx_out_width_over_x_r_pad = (x_padding + input_width + x_stride - 1)/x_stride + 1;
  WORD32 left_shift, right_shift;
  WORD32 out_width_over_x_r_pad = out_width - idx_out_width_over_x_r_pad;

  ae_int32x2 max_int8 = AE_MOVDA32(255);
  ae_int32x2 min_int8 = AE_MOVDA32(0);

  /* When kernel convolves over x-right pad region only, output is just bias */
  for(i = 0; i < out_height; i++)
  {
    for(j = idx_out_width_over_x_r_pad; j < out_width; j++)
    {
      for(k = 0; k < out_channels; k++)
      {
#if TFLITE_SINGLE_ROUNDING
        left_shift  = p_out_shift[k];
        /* Single rounding macro doesn't need two shifts so this is not used */
        (void)right_shift;
#else /* #if TFLITE_SINGLE_ROUNDING */
        left_shift  = p_out_shift[k] < 0 ? 0 : p_out_shift[k];
        right_shift = p_out_shift[k] > 0 ? 0 : -p_out_shift[k];
#endif /* #if TFLITE_SINGLE_ROUNDING */          
        ae_int32x2 acc;
#if XCHAL_HAVE_HIFI1
        if(p_bias != NULL){
           acc = AE_L32_I((ae_int32*)&p_bias[k], 0);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
        AE_S8_0_X_HIFI1( AE_MOVINT16X4_FROMINT32X2(acc), (WORD8 *)p_out, (i * out_height_offset + j * out_width_offset + k * out_channels_offset));
#else
        if(p_bias != NULL){
          acc = AE_MOVDA32(p_bias[k]);
        }
        else{
          acc = AE_MOVDA32(0);
        }
        MPY_BY_QUANT_MULT_X2_OUT32(acc, acc, p_out_multiplier[k], left_shift, right_shift);
        acc = AE_ADD32S(acc, AE_MOVDA32(out_zero_bias));
#if 0
        AE_MINMAX32(acc, min_int8, max_int8);
#else
        acc = AE_MAX32(acc, min_int8);
        acc = AE_MIN32(acc, max_int8);
#endif
        p_out[i * out_height_offset + j * out_width_offset + k * out_channels_offset] = (UWORD8)AE_MOVAD32_L(acc);
#endif
      }
    }
  }
  return out_width_over_x_r_pad;
}

#ifdef polyphase_debug
#include<stdio.h>
void writingoutput(WORD8* __restrict__ p_out_base, WORD32 out_height, WORD32 out_width,WORD32 out_channels )
{
	int i,j, count;
	FILE * dataFilePr;
	count = 0;
	dataFilePr = fopen("C:/Users/hariev/Documents/file.txt", "w+");
	for(i=0;i<out_height;i++)
		for(j=0;j<out_width;j++)
		{
			fprintf(dataFilePr,"%d\n", *(p_out_base+count) );
			count = count + out_channels;
		}
	fclose(dataFilePr);
}
void manipulateinput(void* p_inp, WORD32 input_height, WORD32 input_width, WORD32 input_channels, void* p_ker, WORD32 kernel_height, WORD32 kernel_width, WORD32 output_channels, void* p_bias, WORD32* p_out_multiplier, WORD32* p_out_shift, WORD32* out_zero_bias, WORD32* input_zero_bias)
{
	WORD8* p_inp_debug;
	WORD8* p_ker_debug;
	WORD32* p_bias_debug;

	p_inp_debug  = (WORD8*)p_inp;
	p_ker_debug  = (WORD8*)p_ker;
	p_bias_debug = (WORD32*)p_bias;

	WORD32 iter = 0, i, k, j1, j2;
	for(k=0;k<input_height;k++)
		for(i=0;i<input_width;i++)
		{
			for(j1=0;j1<input_channels;j1++)
			{
				*p_inp_debug = iter;//14*k + 2*i;//iter;
				p_inp_debug++;
			}
			iter++;
			if(iter==8)
				iter = 0;
		}

	for(j2=0;j2<output_channels;j2++)
		for(k=0;k<kernel_height;k++)
			for(i=0;i<kernel_width;i++)
			{
				for(j1=0;j1<input_channels;j1++)
				{

					{
						*p_ker_debug = 1;
						//if( (k==0) && (i==0) && (j2==1))
							//*p_ker_debug = 1;
						p_ker_debug++;
					}
				}
			}

	for(k=0;k<output_channels;k++)
	{
		p_bias_debug[k] = 0;
		p_out_multiplier[k] = 1073741823;//1073741823;///2147483647;
		p_out_shift[k] = -2;
	}

	*out_zero_bias = 0;
	*input_zero_bias = 0;

}
#endif

static void xa_nn_rearrange_chw_to_hwc
              (pWORD8 __restrict__ p_out
              ,const WORD8*  __restrict__ p_inp
              ,WORD32 height
              ,WORD32 width
              ,WORD32 channels
              ) 
{
        const int inp_stride=width*height;
        int itr_ch, itr_h, itr_w;
        for(itr_h = 0; itr_h < height; itr_h++)
        {
          WORD8 *p_inp1 = (WORD8 *) p_inp+(itr_h*width);
          for(itr_w = 0; itr_w < width; itr_w++)
          {
            WORD8 * __restrict__ p_inp2 = p_inp1+(itr_w*1);
            //ae_valign a_out = AE_ZALIGN64();
            for(itr_ch = 0; itr_ch < channels; itr_ch++)
            {

              WORD8 d0 = *(p_inp2);
              p_inp2 += inp_stride;
              *p_out++ = d0;
              
            }
          }
        }

}

WORD32 xa_nn_conv2d_per_chan_asym8xasym8(
    UWORD8* __restrict__ p_out,
    const UWORD8* __restrict__ p_inp,
    const UWORD8* __restrict__ p_kernel,
    const WORD32* __restrict__ p_bias,
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 kernel_channels,
    WORD32 dilation_height,
    WORD32 dilation_width,
    WORD32 out_channels,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 x_padding,
    WORD32 y_padding,
    WORD32 out_height,
    WORD32 out_width,
    WORD32 input_zero_bias,
    WORD32 * p_out_multiplier,
    WORD32 * p_out_shift,
    WORD32 out_zero_bias,
    WORD32 out_data_format,
    VOID *p_scratch)
{
   /* NULL pointer checks */
  XA_NNLIB_ARG_CHK_PTR(p_out, -1);
  XA_NNLIB_ARG_CHK_PTR(p_kernel, -1);
  XA_NNLIB_ARG_CHK_PTR(p_inp, -1);
  XA_NNLIB_ARG_CHK_PTR(p_scratch, -1);
  
  /* Pointer alignment checks */
  XA_NNLIB_ARG_CHK_ALIGN(p_out, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_inp, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_kernel, sizeof(UWORD8), -1);
  XA_NNLIB_ARG_CHK_ALIGN(p_bias, sizeof(WORD32), -1);
  
  //XA_NNLIB_ARG_CHK_ALIGN(p_scratch, sizeof(WORD8), -1);
  /* Basic Parameter checks */
  XA_NNLIB_ARG_CHK_COND((input_height <= 0 || input_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((kernel_height <= 0 || kernel_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_channels <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_stride <= 0 || x_stride <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((y_padding < 0 || x_padding < 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_height <= 0 || out_width <= 0), -1);
  XA_NNLIB_ARG_CHK_COND((input_zero_bias < -255 || input_zero_bias > 0), -1);
  XA_NNLIB_ARG_CHK_COND((out_zero_bias < 0 || out_zero_bias > 255), -1);
  XA_NNLIB_ARG_CHK_COND((out_data_format != 0 && out_data_format != 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_height != 1), -1);
  XA_NNLIB_ARG_CHK_COND((dilation_width != 1), -1);

  int itr;
  for(itr=0;itr<out_channels;itr++){
    XA_NNLIB_ARG_CHK_COND((p_out_shift[itr] < -31 || p_out_shift[itr] > 31), -1);
  }

  const int groups = input_channels/kernel_channels;
  XA_NNLIB_ARG_CHK_COND((groups<=0), -1);
  XA_NNLIB_ARG_CHK_COND(((input_channels %kernel_channels)!=0),-1);
  XA_NNLIB_ARG_CHK_COND(((out_channels%groups)!=0),-1);
  const int kernels_per_group = out_channels / groups;
  XA_NNLIB_ARG_CHK_COND((kernels_per_group<=0),-1);
  
  int ret = 0;

  WORD32 j;
  WORD32 input_bytewidth = 1;
  VOID *pp_inp = (VOID *)p_inp;
  UWORD8* __restrict__ tmp_out;

  p_scratch = ALIGNED_ADDR(p_scratch, ALIGNMENT);
  xa_nn_conv_state_t *p_state = (xa_nn_conv_state_t *)p_scratch;
  WORD32 inp_h, inp_w, ker_h, ker_w, x_str, y_str, x_pad, y_pad, out_h, out_w;
    
  if ((input_height == 1) && (kernel_height == 1) && (out_height == 1))
  {
    inp_h = input_width;
    inp_w = input_height;
    ker_h = kernel_width;
    ker_w = kernel_height;
    x_str = y_stride;
    y_str = x_stride;
    x_pad = y_padding;
    y_pad = x_padding;
    out_h = out_width;
    out_w = out_height;
  }
  else
  {
    inp_h = input_height;
    inp_w = input_width;
    ker_h = kernel_height;
    ker_w = kernel_width;
    x_str = x_stride;
    y_str = y_stride;
    x_pad = x_padding;
    y_pad = y_padding;
    out_h = out_height;
    out_w = out_width;
  }

  WORD32 out_channels_offset = out_data_format ? out_h * out_w : 1;
  WORD32 out_height_offset = out_data_format ? out_w : out_w * out_channels;
  WORD32 out_width_offset = out_data_format ? 1 : out_channels;

  WORD32 x_padding_var = x_pad;
  WORD32 kernel_channels_pad;

  kernel_channels_pad = PADDED_SIZE(kernel_channels, (ALIGNMENT >> 1));

  /* When kernel convolves over x-left pad region only */
  WORD32 out_width_over_x_pad = 0;
  
  if(x_padding_var >= ker_w)
  {
    out_width_over_x_pad = conv_x_left_pad(x_pad, ker_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
    x_padding_var -= out_width_over_x_pad * x_str;
  }

  /* When kernel convolves over x-right pad region only */
  WORD32 out_width_over_x_r_pad = 0;
  // Determine x-right padding
  WORD32 x_r_pad = ker_w + (out_w - 1) * x_str - (x_pad + inp_w);
  x_r_pad = x_r_pad < 0 ? 0 : x_r_pad;
  if(x_r_pad >= ker_w)
  {
    out_width_over_x_r_pad = conv_x_right_pad(x_pad, inp_w, x_str, out_w, out_h, out_channels, out_channels_offset, out_width_offset, out_height_offset, p_bias, p_out, p_out_multiplier, p_out_shift, out_zero_bias);
  }

  /* When kernel convolves over input region */
  p_out += out_width_over_x_pad * out_width_offset;
  // Initialize circular buffer
  // Determine y-bottom padding
  WORD32 y_b_pad = ker_h + (out_h - 1) * y_str - (y_pad + inp_h);
  y_b_pad = y_b_pad < 0 ? 0 : y_b_pad;
  
  xa_nn_conv2d_std_init_state((void*)p_state
      ,(void*)p_kernel
      ,inp_h
      ,input_channels
      ,ker_h
      ,kernel_width
      ,y_str
      ,y_pad
      ,out_h
      ,out_channels
      ,PREC_ASYM8U
      ,PREC_ASYM8U);

  for (int grp_i = 0; grp_i < groups; ++grp_i)
  {
    tmp_out=p_out+grp_i*kernels_per_group*out_channels_offset;
    xa_nn_conv2d_group_init_state((void*)p_state
        ,(void*)p_kernel
        ,inp_h
        ,kernel_channels
        ,ker_h
        ,ker_w
        ,y_str
        ,y_pad
        ,out_h
        ,out_channels
        ,PREC_ASYM8U
        ,PREC_ASYM8U);

    pp_inp = (VOID *)(p_inp+grp_i*kernel_channels);
    
    conv2d_group_init_cir_buf(input_channels, kernel_channels_pad,kernel_channels,input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, p_state, -input_zero_bias);
    
      // Index to padded input width
    WORD32 idx_beg_inp_width_pad = ker_w - x_str;
    idx_beg_inp_width_pad = idx_beg_inp_width_pad < 0 ? 0 : idx_beg_inp_width_pad;
  
    // Process Loop to compute one output plane [out_h x out_channels] per iteration
    for(j=0; j < out_w-out_width_over_x_pad-out_width_over_x_r_pad; j++)
    {
      // Add x_str x (inp_h x input_channels) new planes to circular buffer
      conv2d_group_update_cir_buf(input_channels, kernel_channels_pad,kernel_channels,input_bytewidth, inp_w, inp_h, y_pad, y_b_pad, x_padding_var, ker_w, x_str, (VOID**)&pp_inp, idx_beg_inp_width_pad, p_state, -input_zero_bias);
      
      // Update index to input width padded
      idx_beg_inp_width_pad += x_str;
      
      const WORD32 *p_bias_grp = NULL;
      if(p_bias != NULL){
        p_bias_grp = p_bias+grp_i*kernels_per_group;
      }
      
      xa_nn_matXvec_asym8xasym8_asym8_circ
        (tmp_out /* output */
        ,p_state->cir_buf.p_curr/* matrix: rows x cols */
        ,(p_state->p_kernel_padded+grp_i*kernels_per_group*kernel_channels_pad*ker_w*ker_h) /* vec: cols */
        ,p_bias_grp/* bias */
        ,out_h /* rows */
        ,kernel_channels_pad * ker_w * ker_h /* cols */
        ,kernel_channels_pad * ker_w * y_str/* row_offset */
        ,kernels_per_group /* vec_count */
        ,kernel_channels_pad * ker_w * ker_h /* vec_stride */
        ,out_channels_offset /* out_col_offset */
        ,out_height_offset /* out_row_offset */
        ,input_zero_bias
        ,0
        ,p_out_multiplier[0]
        ,p_out_shift[0]
        ,out_zero_bias
        );

      tmp_out += out_width_offset;
    }
  }
  
  return 0;
}