#include "optimizer.h"

using namespace std;

void ICP_transform(void* source_points_, void* target_points_, void* source_confidence_, void* target_confidence_,
                   void* pair_num_, void* transform_, void* initial_cost_, void* final_cost_) {
    double* source_points = (double*)source_points_;
    double* target_points = (double*)target_points_;
    double* source_confidence = (double*)source_confidence_;
    double* target_confidence = (double*)target_confidence_;
    double* initial_cost = (double*) initial_cost_;
    double* final_cost = (double*) final_cost_;
    int* pair_num = (int*) pair_num_;
    ceres::Problem problem;

    /*Eigen::Matrix3d rotation = Eigen::MatrixXd::Identity(3, 3);
    Sophus::SO3 SO3(rotation);
    Eigen::Vector3d se3 = SO3.log();
    cout<<rotation<<endl;
    cout<<se3<<endl;
    cout<<Sophus::SO3::hat(se3)<<endl;
    cout<<SO3.matrix()<<endl;*/

    double init_trans[6] = {0, 0, 0, 0, 0, 0};
    problem.AddResidualBlock(new ICPResidual(source_points, target_points, source_confidence, target_confidence, *pair_num),
                             nullptr, init_trans); // transform as init_trans
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_type = ceres::LINE_SEARCH;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    *initial_cost = summary.initial_cost;
    *final_cost = summary.final_cost;
    cout<<summary.FullReport()<<endl;
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
//    problem.AddResidualBlock(new SmoothCostFunction(init_points, init_confidence), new ceres::TrivialLoss(), init_points);
//    problem.AddResidualBlock(new GuideCostFunction(guide_points, guide_confidence), new ceres::HuberLoss(0.2), init_points);
    problem.AddResidualBlock(new BoneResidual(bone_length), new ceres::TrivialLoss(), init_points);
    problem.AddResidualBlock(new SpaceResidual(guide_points, init_points, init_confidence, guide_confidence),
                             nullptr, init_points);
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
    // cout<<summary.FullReport()<<endl;
}

