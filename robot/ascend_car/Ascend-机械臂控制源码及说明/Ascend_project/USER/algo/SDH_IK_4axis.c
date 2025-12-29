/*
 * File: SDH_IK_4axis.c
 *
 * MATLAB Coder version            : 5.1
 * C/C++ source code generated on  : 27-Jul-2023 14:56:12
 */

/* Include Files */
#include "SDH_IK_4axis.h"
#include "rt_nonfinite.h"
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Declarations */
static float rt_atan2f_snf(float u0, float u1);

/* Function Definitions */
/*
 * Arguments    : float u0
 *                float u1
 * Return Type  : float
 */
static float rt_atan2f_snf(float u0, float u1)
{
  float y;
  int b_u0;
  int b_u1;
  if (rtIsNaNF(u0) || rtIsNaNF(u1)) {
    y = rtNaNF;
  } else if (rtIsInfF(u0) && rtIsInfF(u1)) {
    if (u0 > 0.0F) {
      b_u0 = 1;
    } else {
      b_u0 = -1;
    }

    if (u1 > 0.0F) {
      b_u1 = 1;
    } else {
      b_u1 = -1;
    }

    y = atan2f((float)b_u0, (float)b_u1);
  } else if (u1 == 0.0F) {
    if (u0 > 0.0F) {
      y = RT_PIF / 2.0F;
    } else if (u0 < 0.0F) {
      y = -(RT_PIF / 2.0F);
    } else {
      y = 0.0F;
    }
  } else {
    y = atan2f(u0, u1);
  }

  return y;
}

/*
 * Arguments    : float x
 *                float y
 *                float z
 *                float R
 *                float P
 *                float Y
 *                float theta[4]
 * Return Type  : void
 */
void SDH_IK_4axis(float x, float y, float z, float R, float P, float Y, float
                  theta[4])
{
  float T[16];
  float tRf[9];
  float C;
  float ct_idx_0;
  float ct_idx_1;
  float ct_idx_2;
  float st_idx_0;
  float st_idx_1;
  float st_idx_2;
  float tRf_tmp;
  int T_tmp;
  int i;
  st_idx_0 = Y * 3.14159274F / 180.0F;
  st_idx_1 = P * 3.14159274F / 180.0F;
  st_idx_2 = R * 3.14159274F / 180.0F;
  ct_idx_0 = cosf(st_idx_0);
  st_idx_0 = sinf(st_idx_0);
  ct_idx_1 = cosf(st_idx_1);
  st_idx_1 = sinf(st_idx_1);
  ct_idx_2 = cosf(st_idx_2);
  st_idx_2 = sinf(st_idx_2);
  tRf[0] = ct_idx_1 * ct_idx_2;
  tRf[3] = -ct_idx_1 * st_idx_2;
  tRf[6] = st_idx_1;
  tRf_tmp = ct_idx_2 * st_idx_0;
  tRf[1] = ct_idx_0 * st_idx_2 + tRf_tmp * st_idx_1;
  C = ct_idx_0 * ct_idx_2;
  tRf[4] = C - st_idx_0 * st_idx_1 * st_idx_2;
  tRf[7] = -ct_idx_1 * st_idx_0;
  tRf[2] = st_idx_0 * st_idx_2 - C * st_idx_1;
  tRf[5] = tRf_tmp + ct_idx_0 * st_idx_1 * st_idx_2;
  tRf[8] = ct_idx_0 * ct_idx_1;
  for (i = 0; i < 3; i++) {
    T_tmp = i << 2;
    T[T_tmp] = tRf[3 * i];
    T[T_tmp + 1] = tRf[3 * i + 1];
    T[T_tmp + 2] = tRf[3 * i + 2];
  }

  ct_idx_1 = rt_atan2f_snf(-T[8], T[9]) + 3.14159274F;

  /* theta1=atan2(-ax,ay); */
  st_idx_2 = x * cosf(ct_idx_1) + y * sinf(ct_idx_1);
  C = st_idx_2 - 161.77F * T[6];
  ct_idx_2 = (z - 5.5F) - 161.77F * T[2];
  ct_idx_0 = acosf((((C * C + ct_idx_2 * ct_idx_2) - 7841.10254F) - 10816.0F) /
                   18418.4F);
  st_idx_0 = -33648.1602F * sinf(ct_idx_0);
  ct_idx_2 = cosf(ct_idx_0);
  st_idx_1 = 33648.1602F * ct_idx_2 + 28649.4668F;
  C = -(((((st_idx_2 * st_idx_2 + (z - 5.5F) * (z - 5.5F)) - 10816.0F) -
          7841.10254F) - 26169.5332F) - 18418.4F * ct_idx_2);
  C = -rt_atan2f_snf(st_idx_1, st_idx_0) + rt_atan2f_snf(-C, -sqrtf((st_idx_0 *
    st_idx_0 + st_idx_1 * st_idx_1) - C * C));

  /* theta234=atan2(ox*cos(theta1)+oy*sin(theta1),-oz); */
  /*  angle_limit = [-90 90; -90 90; -90 90; -90 90]; */
  theta[0] = 57.2957802F * ct_idx_1;
  theta[1] = 57.2957802F * ((rt_atan2f_snf(T[2], T[6]) - C) - ct_idx_0) - 90.0F;
  theta[2] = 57.2957802F * ct_idx_0;
  theta[3] = 57.2957802F * C;

  /*  for i = 1:4 */
  /*      if theta(i) > angle_limit(i, 2) */
  /*          theta(i) = angle_limit(i, 2); */
  /*      end */
  /*      if theta(i) < angle_limit(i, 1) */
  /*          theta(i) = angle_limit(i, 1); */
  /*      end */
  /*  end */
  /*   */
  /*   */
  /*  Theta = theta; */
}

/*
 * File trailer for SDH_IK_4axis.c
 *
 * [EOF]
 */
