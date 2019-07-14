#include "optimizer.h"
#include <iostream>

using namespace std;

void guide_transform(void* last_points_, void* guide_points_, void* guide_confidence_,
                     void* init_trans_, void* initial_cost_, void* final_cost_){
    double* last_points = (double*)last_points_;
    double* guide_points = (double*)guide_points_;
    double* guide_confidence = (double*)guide_confidence_;
    double* init_trans = (double*)init_trans_;
    double* initial_cost = (double*) initial_cost_;
    double* final_cost = (double*) final_cost_;
    ceres::Problem problem;
    problem.AddResidualBlock(new TransformResidual(last_points, guide_points, guide_confidence, init_trans),
                             nullptr, init_trans);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    *initial_cost = summary.initial_cost;
    *final_cost = summary.final_cost;
    //cout<<summary.FullReport()<<endl;
}

void refine_skeleton(void* init_points_, void* guide_points_, void* init_confidence_, void* guide_confidence_,
                     void* bone_length_, void* initial_cost_, void* final_cost_){
    double* init_points = (double*)init_points_;
    double* guide_points = (double*)guide_points_;
    double* init_confidence = (double*)init_confidence_;
    double* guide_confidence = (double*)guide_confidence_;
    double* bone_length = (double*) bone_length_;
    double* initial_cost = (double*) initial_cost_;
    double* final_cost = (double*) final_cost_;
    ceres::Problem problem;
    problem.AddResidualBlock(new SpaceResidual(guide_points, init_points, init_confidence, guide_confidence),
                             nullptr, init_points);
    /*for(int i = 0; i < 14; ++i) {
        problem.AddResidualBlock(new SmoothCostFunction(init_points, init_confidence[i], i), nullptr, init_points);
        problem.AddResidualBlock(new GuideCostFunction(guide_points, guide_confidence[i], i), new ceres::HuberLoss(10), init_points);
    }*/
    problem.AddResidualBlock(new BoneResidual(bone_length), nullptr, init_points);
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::LINE_SEARCH;//TRUST_REGION;//
    options.minimizer_progress_to_stdout = false;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    *initial_cost = summary.initial_cost;
    *final_cost = summary.final_cost;
    //cout<<summary.FullReport()<<endl;
}

