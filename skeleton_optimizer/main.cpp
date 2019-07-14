#include <iostream>
#include <ceres/ceres.h>
#include "optimizer.h"

int main() {
    double bone_length[13] = {0.20920274, 0.19895618, 0.19658908, 0.5948734, 0.5993103, 0.26214048,
                              0.22161089, 0.2478241, 0.223972, 0.35695255, 0.26643517, 0.36374225, 0.28344997};
    double init_points[42] = {3.18760521e-01,  -7.23749947e-01,   1.28363562e+00,
                              3.31803407e-01, -5.15769631e-01,  1.31450123e+00,
                              4.88228930e-01, -3.91726179e-01,  1.27895387e+00,
                              1.44270142e-01, -4.41724429e-01,  1.33239520e+00,
                              5.54646869e-01, -1.35190803e-01,  1.27935735e+00,
                              9.41235813e-02, -2.10544010e-01,  1.41253902e+00,
                              5.64586033e-01,  6.61712099e-02,  1.17874958e+00,
                              -2.40139186e-01,  7.55379945e-03,  1.40910247e+00,
                              2.92889731e-01,  6.88367545e-02,  1.24467648e+00,
                              1.29996085e-01,  4.54479119e-02,  1.27517116e+00,
                              3.14493904e-01,  4.22639331e-01,  1.22743227e+00,
                              -1.32490604e-06,  3.75163772e-01,  1.27562230e+00,
                              2.83937970e-01,  6.63613942e-01,  1.33334241e+00,
                              -4.01451580e-03,  6.69195062e-01,  1.37911122e+00};
    double guide_points[42] = {0.3+3.18760521e-01,  -7.23749947e-01,0.1+   1.28363562e+00,
                               0.3+3.31803407e-01, -5.15769631e-01, 0.1+ 1.31450123e+00,
                               0.3+4.88228930e-01, -3.91726179e-01, 0.1+ 1.27895387e+00,
                               0.3+1.44270142e-01, -4.41724429e-01, 0.1+ 1.33239520e+00,
                               0.3+5.54646869e-01, -1.35190803e-01, 0.1+ 1.27935735e+00,
                               0.3+9.41235813e-02, -2.10544010e-01, 0.1+ 1.41253902e+00,
                               0.3+5.64586033e-01,  6.61712099e-02, 0.1+ 1.17874958e+00,
                               0.3+-2.40139186e-01,  7.55379945e-03,0.1+  1.40910247e+00,
                               0.3+2.92889731e-01,  6.88367545e-02, 0.1+ 1.24467648e+00,
                               0.3+1.29996085e-01,  4.54479119e-02, 0.1+ 1.27517116e+00,
                               0.3+3.14493904e-01,  4.22639331e-01, 0.1+ 1.22743227e+00,
                               0.3+-1.32490604e-06,  3.75163772e-01,0.1+  1.27562230e+00,
                               0.3+2.83937970e-01,  6.63613942e-01, 0.1+ 1.33334241e+00,
                               0.3+-4.01451580e-03,  6.69195062e-01,0.1+  1.37911122e+00};
    double confidence[14] = {};
    std::fill(confidence, confidence + 14, 1.0);

    double temp = 0.0;
    double* init_cost = &temp;
    double* final_cost = &temp;

    double init_trans[3] = {-0.1, 0, 0};
    /*guide_transform(init_points, guide_points, confidence, init_trans, init_cost, final_cost);
    for(int i = 0; i < 3; ++i){
        cout<<init_trans[i]<<endl;
    }*/

    refine_skeleton(init_points, guide_points, confidence, confidence, bone_length, init_cost, final_cost);
    for(int i = 0; i < 14; ++i){
        for(int j = 0; j < 3; ++j)
            cout<<init_points[i * 3 + j] - guide_points[i * 3 + j]<<'\t';
        cout<<endl;
    }

    return 0;
}
