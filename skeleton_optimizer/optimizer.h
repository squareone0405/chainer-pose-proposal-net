#ifndef SKELETON_OPTIMIZER_OPTIMIZER_H
#define SKELETON_OPTIMIZER_OPTIMIZER_H

#include <iostream>
#include <ceres/ceres.h>

using namespace std;

class SmoothCostFunction : public ceres::SizedCostFunction<1, 42> {
private:
    double init_confidence;
    double init_points[42];
    const double alpha_init = 200;
    int point_idx;
    double epsilon = 1e-10;

public:
    SmoothCostFunction(double* init_points, double init_confidence, int point_idx) {
        memcpy(this->init_points, init_points, sizeof(double) * 42);
        this->init_confidence = init_confidence;
        this->point_idx = point_idx;
    }

    virtual ~SmoothCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        residuals[0] = 0.0;
        double dist = 0.0;
        for(int i = 0; i < 3; ++i)
            dist += (parameters[0][3 * point_idx + i] - init_points[3 * point_idx + i]) *
                    (parameters[0][3 * point_idx + i] - init_points[3 * point_idx + i]);
        residuals[0] = alpha_init * this->init_confidence * sqrt(dist + epsilon);

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 42; ++i)
            jacobians[0][i] = 0.0;

        for(int i = 0; i < 3; ++i){
            jacobians[0][3 * point_idx + i] = alpha_init * this->init_confidence *
                                              (parameters[0][3 * point_idx + i] - init_points[3 * point_idx + i]) / sqrt(dist + epsilon);
        }

        return true;
    }
};

class GuideCostFunction : public ceres::SizedCostFunction<1, 42> {
private:
    double guide_confidence;
    double guide_points[42];
    const double alpha_guide = 100;
    int point_idx;
    double epsilon = 1e-10;

public:
    GuideCostFunction(double* guide_points, double guide_confidence, int point_idx) {
        memcpy(this->guide_points, guide_points, sizeof(double) * 42);
        this->guide_confidence = guide_confidence;
        this->point_idx = point_idx;
    }

    virtual ~GuideCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        residuals[0] = 0.0;
        double dist = 0.0;
        for(int i = 0; i < 3; ++i)
            dist += (parameters[0][3 * point_idx + i] - guide_points[3 * point_idx + i]) *
                    (parameters[0][3 * point_idx + i] - guide_points[3 * point_idx + i]);
        residuals[0] = alpha_guide * this->guide_confidence * sqrt(dist + epsilon);

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 42; ++i)
            jacobians[0][i] = 0.0;

        for(int i = 0; i < 3; ++i){
            jacobians[0][3 * point_idx + i] = alpha_guide * this->guide_confidence *
                                              (parameters[0][3 * point_idx + i] - guide_points[3 * point_idx + i]) / sqrt(dist + epsilon);
        }

        return true;
    }
};

class SpaceResidual : public ceres::SizedCostFunction<1, 42> {
private:
    double* guide_points;
    double* init_confidence;
    double* guide_confidence;
    double init_points[42];
    const double alpha_guide = 50;
    const double alpha_init = 100;
    double epsilon = 1e-6;
    const double huber_thres = 0.01; // for guide points
    const double tukey_thres = 0.4; // for guide points

private:
    double get_huber_loss_diff(double x) const {
        if(fabs(x) > huber_thres)
            if(x > 0)
                return 2 * huber_thres;
            else
                return -2 * huber_thres;
        else
            return 2 * x;
    }
    double get_huber_loss(double x) const {
        if(fabs(x) > huber_thres)
            return huber_thres * (2 * fabs(x) - huber_thres);
        else
            return x * x;
    }
    double get_tukey_loss_diff(double x) const {
        if(fabs(x) > tukey_thres)
            return 0;
        else
            return x * pow((1 - (x * x)/(tukey_thres * tukey_thres)), 2);
    }
    double get_tukey_loss(double x) const {
        return (x * x) * (3 * pow(tukey_thres, 4) - 3 * tukey_thres * tukey_thres * x * x + pow(x, 4))
               / (4 * pow(tukey_thres, 4));
    }

public:
    SpaceResidual(double* guide_points, double* init_points, double* init_confidence, double* guide_confidence){
        this->guide_points = guide_points;
        memcpy(this->init_points, init_points, sizeof(double) * 42);
        this->init_confidence = init_confidence;
        this->guide_confidence = guide_confidence;
    }

    virtual ~SpaceResidual() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        residuals[0] = 0.0;
        double dist_guide[14] = {0.0};
        double dist_init[14] = {0.0};
        for(int i = 0; i < 14; ++i){
            if(guide_points[3 * i + 2] > epsilon) {
                for(int j = 0; j < 3; ++j) {
                    dist_guide[i] += (parameters[0][3 * i + j] - guide_points[3 * i + j]) *
                                     (parameters[0][3 * i + j] - guide_points[3 * i + j]);
                }
                dist_guide[i] = dist_guide[i] + epsilon;
                residuals[0] += alpha_guide * this->guide_confidence[i] * get_huber_loss(dist_guide[i]);
            }
        }
        for(int i = 0; i < 14; ++i) {
            for(int j = 0; j < 3; ++j) {
                dist_init[i] += (parameters[0][3 * i + j] - init_points[3 * i + j]) *
                                (parameters[0][3 * i + j] - init_points[3 * i + j]);
            }
            dist_init[i] = dist_init[i] + epsilon;
            residuals[0] += alpha_init * this->init_confidence[i] * dist_init[i];
        }

        //cout<<"space residual: "<<residuals[0]<<endl;

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 42; ++i)
            jacobians[0][i] = 0.0;


        for(int i = 0; i < 14; ++i){
            if(guide_points[3 * i + 2] > epsilon){
                for(int j = 0; j < 3; ++j) {
                    jacobians[0][3 * i + j] += alpha_guide * this->guide_confidence[i] *
                                               (parameters[0][3 * i + j] - guide_points[3 * i + j]) *
                                               get_huber_loss_diff(dist_guide[i]);
                }
            }
        }

        for(int i = 0; i < 42; ++i){
            jacobians[0][i] += this->init_confidence[i / 3] * alpha_init *
                               (parameters[0][i] - init_points[i]);
        }

        return true;
    }
};

class BoneResidual : public ceres::SizedCostFunction<1, 42> {
private:
    double* bone_length;
    const int edges_begin[13] = {1, 1, 1, 1, 1, 2, 4, 3, 5, 8, 10, 9, 11};
    const int edges_end[13] = {0, 2, 3, 8, 9, 4, 6, 5, 7, 10, 12, 11, 13};
    const double beta = 300;

public:
    BoneResidual(double* bone_length){
        this->bone_length = bone_length;
    }

    virtual ~BoneResidual() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        residuals[0] = 0.0;

        double distance[13] = {};

        for(int i = 0; i < 13; ++i){
            int begin = edges_begin[i];
            int end = edges_end[i];
            distance[i] = sqrt((parameters[0][begin * 3] - parameters[0][end * 3]) *
                               (parameters[0][begin * 3] - parameters[0][end * 3]) +
                               (parameters[0][begin * 3 + 1] - parameters[0][end * 3 + 1]) *
                               (parameters[0][begin * 3 + 1] - parameters[0][end * 3 + 1]) +
                               (parameters[0][begin * 3 + 2] - parameters[0][end * 3 + 2]) *
                               (parameters[0][begin * 3 + 2] - parameters[0][end * 3 + 2]));
        }

        for(int i = 0; i < 13; ++i){
            residuals[0] += (distance[i] - bone_length[i]) * (distance[i] - bone_length[i]);
        }

        //cout<<"bone residual:"<<residuals[0]<<endl;

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 42; ++i)
            jacobians[0][i] = 0.0;

        for(int i = 0; i < 13; ++i){
            int begin = edges_begin[i];
            int end = edges_end[i];
            for(int j = 0; j < 3; ++j) {
                double diff = beta * (distance[i] - bone_length[i]) *
                              (parameters[0][begin * 3 + j] - parameters[0][end * 3 + j]) / distance[i];
                jacobians[0][begin * 3 + j] += diff;
                jacobians[0][end * 3 + j] -= diff;
            }
        }

        return true;
    }
};

class TransformResidual : public ceres::SizedCostFunction<1, 3> {
private:
    double* guide_points;
    double* last_points;
    double* guide_confidence;
    double trans_init[3];
    const double alpha_guide = 100.0;
    const double alpha_trans = 100.0;
    double epsilon = 1e-10;
    const double huber_thres = 0.1;
    const double tukey_thres = 0.4;

private:
    double get_huber_loss_diff(double x) const {
        if(fabs(x) > huber_thres)
            if(x > 0)
                return 2 * huber_thres;
            else
                return -2 * huber_thres;
        else
            return 2 * x;
    }
    double get_huber_loss(double x) const {
        if(fabs(x) > huber_thres)
            return huber_thres * (2 * fabs(x) - huber_thres);
        else
            return x * x;
    }

    double get_tukey_loss_diff(double x) const {
        if(fabs(x) > tukey_thres)
            return 0;
        else
            return x * pow((1 - (x * x)/(tukey_thres * tukey_thres)), 2);
    }
    double get_tukey_loss(double x) const {
        return (x * x) * (3 * pow(tukey_thres, 4) - 3 * tukey_thres * tukey_thres * x * x + pow(x, 4))
               / (4 * pow(tukey_thres, 4));
    }

public:
    TransformResidual(double* last_points, double* guide_points, double* guide_confidence, double* trans_init){
        this->guide_points = guide_points;
        this->last_points = last_points;
        this->guide_confidence = guide_confidence;
        memcpy(this->trans_init, trans_init, sizeof(double) * 3);
    }

    virtual ~TransformResidual() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {

        residuals[0] = 0.0;
        double dist[14];
        for(int i = 0; i < 14; ++i){
            dist[i] = 0.0;
            if(guide_points[3 * i + 2] > epsilon) {
                for(int j = 0; j < 3; ++j){
                    dist[i] += (guide_points[3 * i + j] + parameters[0][j] - last_points[3 * i + j]) *
                               (guide_points[3 * i + j] + parameters[0][j] - last_points[3 * i + j]);
                }
                dist[i] = sqrt(dist[i]);
                residuals[0] += alpha_guide * this->guide_confidence[i] * get_huber_loss(dist[i]);
            }
        }
        for(int i = 0; i < 3; ++i){
            residuals[0] += alpha_trans * parameters[0][i] * parameters[0][i];
        }

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 3; ++i)
            jacobians[0][i] = 0.0;

        for(int i = 0; i < 14; ++i){
            if(guide_points[3 * i + 2] > epsilon){
                for(int j = 0; j < 3; ++j){
                    jacobians[0][j] += alpha_guide * this->guide_confidence[i] * get_huber_loss_diff(dist[i])
                                       * (guide_points[3 * i + j] + parameters[0][j] - last_points[3 * i + j]);
                }
            }
        }
        for(int i = 0; i < 3; ++i) {
            jacobians[0][i] += alpha_trans * (parameters[0][i] - this->trans_init[i]);
        }
        return true;
    }
};

extern "C" void refine_skeleton(void* init_points_, void* guide_points_, void* init_confidence_, void* guide_confidence_,
                                void* bone_length_, void* initial_cost_, void* final_cost_);

extern "C" void guide_transform(void* last_points_, void* guide_points_, void* guide_confidence_,
                                void* init_trans_, void* initial_cost_, void* final_cost_);

#endif //SKELETON_OPTIMIZER_OPTIMIZER_H
