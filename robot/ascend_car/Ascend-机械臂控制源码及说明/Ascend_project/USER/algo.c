//
// Created by lenovo on 2023/7/14.
//
#include "include.h"

void AlgoInverseKinematics(float pose[6], float angle[4]){  //逆解算法

    SDH_IK_4axis(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5], angle);
}

void AlgoPositiveSolution(float theta[4], float T04[16], float Pos[6]){    //正解算法

    four_Forward_standardDH(theta,T04,Pos);	
}


