//
// Created by lenovo on 2023/7/14.
//

#ifndef HAUWEI_SHENGTENG_PROJECT_ALGO_H
#define HAUWEI_SHENGTENG_PROJECT_ALGO_H


#include "four_Forward_standardDH.h"
#include "SDH_IK_4axis.h"

void AlgoInverseKinematics(float pose[6], float angle[4]);
void AlgoPositiveSolution(float theta[4], float T04[16], float Pos[6]);

#endif //HAUWEI_SHENGTENG_PROJECT_ALGO_H
