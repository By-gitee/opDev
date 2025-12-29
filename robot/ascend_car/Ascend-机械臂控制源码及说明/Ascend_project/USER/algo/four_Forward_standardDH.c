/*
 * File: four_Forward_standardDH.c
 *
 * MATLAB Coder version            : 5.1
 * C/C++ source code generated on  : 27-Jul-2023 15:19:18
 */

/* Include Files */
#include "four_Forward_standardDH.h"
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
 * 正向运动学求解程序
 * Arguments    : float theta[4]
 *                float T04[16]
 *                float Pos[6]
 * Return Type  : void
 */
void four_Forward_standardDH(float theta[4], float T04[16], float Pos[6])
{
  double rpy_idx_0;
  double rpy_idx_1;
  double rpy_idx_2;
  float c_T01_tmp[16];
  float c_T12_tmp[16];
  float d_T01_tmp[16];
  float T01_tmp;
  float T12_tmp;
  float T23_tmp;
  float T34_tmp;
  float b_T01_tmp;
  float b_T12_tmp;
  float b_T23_tmp;
  float b_T34_tmp;
  int i;
  int i1;
  int i2;

  /* ************************************************************************* */
  theta[0] = theta[0] * 3.14159274F / 180.0F;
  theta[1] = (theta[1] + 90.0F) * 3.14159274F / 180.0F;
  theta[2] = theta[2]  * 3.14159274F / 180.0F;
  theta[3] = theta[3] * 3.14159274F / 180.0F;

  /* theta关节角，offset关节旋转角 */
  /*  求齐次变换矩阵 */
  T01_tmp = sinf(theta[0]);
  b_T01_tmp = cosf(theta[0]);
  T12_tmp = sinf(theta[1]);
  b_T12_tmp = cosf(theta[1]);
  T23_tmp = sinf(theta[2]);
  b_T23_tmp = cosf(theta[2]);
  T34_tmp = sinf(theta[3]);
  b_T34_tmp = cosf(theta[3]);

  /*  T45=[ cos(theta5),-sin(theta5)*cos(alp5), sin(theta5)*sin(alp5),a5*cos(theta5); */
  /*          sin(theta5), cos(theta5)*cos(alp5),-cos(theta5)*sin(alp5),a5*sin(theta5); */
  /*                0,           sin(alp5),           cos(alp5),          d5; */
  /*                0,                   0,                   0,                  1 */
  /*      ]; */
  c_T01_tmp[0] = b_T01_tmp;
  c_T01_tmp[4] = -T01_tmp * 6.12323426E-17F;
  c_T01_tmp[8] = T01_tmp;
  c_T01_tmp[12] = 0.0F * b_T01_tmp;
  c_T01_tmp[1] = T01_tmp;
  c_T01_tmp[5] = b_T01_tmp * 6.12323426E-17F;
  c_T01_tmp[9] = -b_T01_tmp;
  c_T01_tmp[13] = 0.0F * T01_tmp;
  c_T12_tmp[0] = b_T12_tmp;
  c_T12_tmp[4] = -T12_tmp;
  c_T12_tmp[8] = T12_tmp * 0.0F;
  c_T12_tmp[12] = 104.0F * b_T12_tmp;
  c_T12_tmp[1] = T12_tmp;
  c_T12_tmp[5] = b_T12_tmp;
  c_T12_tmp[9] = -b_T12_tmp * 0.0F;
  c_T12_tmp[13] = 104.0F * T12_tmp;
  c_T01_tmp[2] = 0.0F;
  c_T01_tmp[3] = 0.0F;
  c_T12_tmp[2] = 0.0F;
  c_T12_tmp[3] = 0.0F;
  c_T01_tmp[6] = 1.0F;
  c_T01_tmp[7] = 0.0F;
  c_T12_tmp[6] = 0.0F;
  c_T12_tmp[7] = 0.0F;
  c_T01_tmp[10] = 6.12323426E-17F;
  c_T01_tmp[11] = 0.0F;
  c_T12_tmp[10] = 1.0F;
  c_T12_tmp[11] = 0.0F;
  c_T01_tmp[14] = 5.5F;
  c_T01_tmp[15] = 1.0F;
  c_T12_tmp[14] = 0.0F;
  c_T12_tmp[15] = 1.0F;
  for (i = 0; i < 4; i++) {
    T01_tmp = c_T01_tmp[i];
    b_T01_tmp = c_T01_tmp[i + 4];
    T12_tmp = c_T01_tmp[i + 8];
    b_T12_tmp = c_T01_tmp[i + 12];
    for (i1 = 0; i1 < 4; i1++) {
      i2 = i1 << 2;
      d_T01_tmp[i + i2] = ((T01_tmp * c_T12_tmp[i2] + b_T01_tmp * c_T12_tmp[i2 +
                            1]) + T12_tmp * c_T12_tmp[i2 + 2]) + b_T12_tmp *
        c_T12_tmp[i2 + 3];
    }
  }

  c_T12_tmp[0] = b_T23_tmp;
  c_T12_tmp[4] = -T23_tmp;
  c_T12_tmp[8] = T23_tmp * 0.0F;
  c_T12_tmp[12] = 88.55F * b_T23_tmp;
  c_T12_tmp[1] = T23_tmp;
  c_T12_tmp[5] = b_T23_tmp;
  c_T12_tmp[9] = -b_T23_tmp * 0.0F;
  c_T12_tmp[13] = 88.55F * T23_tmp;
  c_T12_tmp[2] = 0.0F;
  c_T12_tmp[3] = 0.0F;
  c_T12_tmp[6] = 0.0F;
  c_T12_tmp[7] = 0.0F;
  c_T12_tmp[10] = 1.0F;
  c_T12_tmp[11] = 0.0F;
  c_T12_tmp[14] = 0.0F;
  c_T12_tmp[15] = 1.0F;
  for (i = 0; i < 4; i++) {
    T01_tmp = d_T01_tmp[i];
    b_T01_tmp = d_T01_tmp[i + 4];
    T12_tmp = d_T01_tmp[i + 8];
    b_T12_tmp = d_T01_tmp[i + 12];
    for (i1 = 0; i1 < 4; i1++) {
      i2 = i1 << 2;
      c_T01_tmp[i + i2] = ((T01_tmp * c_T12_tmp[i2] + b_T01_tmp * c_T12_tmp[i2 +
                            1]) + T12_tmp * c_T12_tmp[i2 + 2]) + b_T12_tmp *
        c_T12_tmp[i2 + 3];
    }
  }

  c_T12_tmp[0] = b_T34_tmp;
  c_T12_tmp[4] = -T34_tmp;
  c_T12_tmp[8] = T34_tmp * 0.0F;
  c_T12_tmp[12] = 161.77F * b_T34_tmp;
  c_T12_tmp[1] = T34_tmp;
  c_T12_tmp[5] = b_T34_tmp;
  c_T12_tmp[9] = -b_T34_tmp * 0.0F;
  c_T12_tmp[13] = 161.77F * T34_tmp;
  c_T12_tmp[2] = 0.0F;
  c_T12_tmp[3] = 0.0F;
  c_T12_tmp[6] = 0.0F;
  c_T12_tmp[7] = 0.0F;
  c_T12_tmp[10] = 1.0F;
  c_T12_tmp[11] = 0.0F;
  c_T12_tmp[14] = 0.0F;
  c_T12_tmp[15] = 1.0F;
  for (i = 0; i < 4; i++) {
    T01_tmp = c_T01_tmp[i];
    b_T01_tmp = c_T01_tmp[i + 4];
    T12_tmp = c_T01_tmp[i + 8];
    b_T12_tmp = c_T01_tmp[i + 12];
    for (i1 = 0; i1 < 4; i1++) {
      i2 = i1 << 2;
      T04[i + i2] = ((T01_tmp * c_T12_tmp[i2] + b_T01_tmp * c_T12_tmp[i2 + 1]) +
                     T12_tmp * c_T12_tmp[i2 + 2]) + b_T12_tmp * c_T12_tmp[i2 + 3];
    }
  }

  /*  求末端位置 */
  /* 采取T04前三行第四列元素作为末端的位置 */
  if (fabsf(fabsf(T04[8]) - 1.0F) < 2.22044605E-16F) {
    /*  when |R13| == 1 */
    /*  singularity */
    rpy_idx_0 = 0.0;

    /*  roll is zero */
    if (T04[8] > 0.0F) {
      rpy_idx_2 = rt_atan2f_snf(T04[6], T04[5]);

      /*  R+Y */
    } else {
      rpy_idx_2 = -rt_atan2f_snf(T04[1], T04[2]);

      /*  R-Y */
    }

    rpy_idx_1 = asinf(T04[8]);
  } else {
    rpy_idx_0 = -rt_atan2f_snf(T04[4], T04[0]);
    rpy_idx_2 = -rt_atan2f_snf(T04[9], T04[10]);
    rpy_idx_1 = atanf(T04[8] * (float)cos(rpy_idx_0) / T04[0]);
  }

  rpy_idx_0 = rpy_idx_0 * 180.0 / 3.1415926535897931;
  rpy_idx_1 = rpy_idx_1 * 180.0 / 3.1415926535897931;
  rpy_idx_2 = rpy_idx_2 * 180.0 / 3.1415926535897931;

  /* 弧度制转为角度制 */
  Pos[0] = T04[12];
  Pos[1] = T04[13];
  Pos[2] = T04[14];
  Pos[3] = (float)rpy_idx_0;
  Pos[4] = (float)rpy_idx_1;
  Pos[5] = (float)rpy_idx_2;

  /* {roll,pitch,yaw}分别表示的是绕{x,y,z}轴的旋转角度 */
}

/*
 * File trailer for four_Forward_standardDH.c
 *
 * [EOF]
 */
