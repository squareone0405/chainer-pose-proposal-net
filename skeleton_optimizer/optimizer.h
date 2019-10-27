#ifndef SKELETON_OPTIMIZER_OPTIMIZER_H
#define SKELETON_OPTIMIZER_OPTIMIZER_H

#include <iostream>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.h>
#include <sophus/so3.h>

typedef Eigen::Matrix<double,6,1> Vector6d;

using namespace std;

class SmoothCostFunction : public ceres::SizedCostFunction<14, 42> {
private:
    double* init_confidence;
    double init_points[42];
    const double alpha_init = 4;
    double epsilon = 1e-10;

public:
    SmoothCostFunction(double* init_points, double* init_confidence) {
        memcpy(this->init_points, init_points, sizeof(double) * 42);
        this->init_confidence = init_confidence;
    }

    virtual ~SmoothCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double dist[14] = {0.0};
        for(int i = 0; i < 14; ++i) {
            for(int j = 0; j < 3; ++j)
                dist[i] += (parameters[0][3 * i + j] - init_points[3 * i + j]) *
                           (parameters[0][3 * i + j] - init_points[3 * i + j]);
            dist[i] = sqrt(dist[i] + epsilon);
            residuals[i] = alpha_init * this->init_confidence[i] * dist[i];
        }

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 14; ++i) {
            for(int j = 0; j < 42; ++j) {
                if(3 * i <= j && j < 3 * i + 3)
                    jacobians[0][42 * i + j] = alpha_init * this->init_confidence[i] *
                                               (parameters[0][j] - init_points[j]) / dist[i];
                else
                    jacobians[0][42 * i + j] = 0.0;
            }
        }

        return true;
    }
};

class GuideCostFunction : public ceres::SizedCostFunction<14, 42> {
private:
    double* guide_confidence;
    double guide_points[42];
    const double alpha_guide = 1;
    double epsilon = 1e-10;

public:
    GuideCostFunction(double* guide_points, double* guide_confidence) {
        memcpy(this->guide_points, guide_points, sizeof(double) * 42);
        this->guide_confidence = guide_confidence;
    }

    virtual ~GuideCostFunction() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double dist[14] = {0.0};
        for(int i = 0; i < 14; ++i) {
            for (int j = 0; j < 3; ++j){
                dist[i] += (parameters[0][3 * i + j] - guide_points[3 * i + j]) *
                           (parameters[0][3 * i + j] - guide_points[3 * i + j]);
            }
            dist[i] = sqrt(dist[i] + epsilon);
            residuals[i] = alpha_guide * this->guide_confidence[i] * dist[i];
        }

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 14; ++i) {
            for(int j = 0; j < 42; ++j) {
                if(3 * i <= j && j < 3 * i + 3)
                    jacobians[0][42 * i + j] = alpha_guide * this->guide_confidence[i] *
                                               (parameters[0][j] - guide_points[j]) / dist[i];
                else
                    jacobians[0][42 * i + j] = 0.0;
            }
        }

        return true;
    }
};

/*class BoneResidual : public ceres::SizedCostFunction<13, 42> {
private:
    double* bone_length;
    const int edges_begin[13] = {1, 1, 1, 1, 1, 2, 4, 3, 5, 8, 10, 9, 11};
    const int edges_end[13] = {0, 2, 3, 8, 9, 4, 6, 5, 7, 10, 12, 11, 13};
    const double beta = 3;
    double epsilon = 1e-10;

public:
    BoneResidual(double* bone_length){
        this->bone_length = bone_length;
    }

    virtual ~BoneResidual() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const {
        double distance[13] = {0.0};
        double loss = 0.0;
        for(int i = 0; i < 13; ++i) {
            int begin = edges_begin[i];
            int end = edges_end[i];
            distance[i] = sqrt((parameters[0][begin * 3] - parameters[0][end * 3]) *
                               (parameters[0][begin * 3] - parameters[0][end * 3]) +
                               (parameters[0][begin * 3 + 1] - parameters[0][end * 3 + 1]) *
                               (parameters[0][begin * 3 + 1] - parameters[0][end * 3 + 1]) +
                               (parameters[0][begin * 3 + 2] - parameters[0][end * 3 + 2]) *
                               (parameters[0][begin * 3 + 2] - parameters[0][end * 3 + 2]) + epsilon);
            residuals[i] = beta * distance[i] - bone_length[i];
            loss += residuals[i] * residuals[i];
        }
        // cout<<loss<<endl;

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]) {
            return true;
        }

        for(int i = 0; i < 13; ++i) {
            int begin = edges_begin[i];
            int end = edges_end[i];
            for(int j = 0; j < 42; ++j) {
                double diff = beta * (parameters[0][begin * 3 + j % 3] - parameters[0][end * 3 + j % 3]) / distance[i];
                if(3 * begin <= j && j < 3 * begin + 3) {
                    jacobians[0][42 * i + j] = diff;
                }
                else if(3 * end <= j && j < 3 * end + 3) {
                    jacobians[0][42 * i + j] = -1.0 * diff;
                }
                else
                    jacobians[0][42 * i + j] = 0.0;
            }
        }
        return true;
    }
};*/

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
            residuals[0] += beta * (distance[i] - bone_length[i]) * (distance[i] - bone_length[i]);
        }

        // cout<<"bone residual:"<<residuals[0]<<endl;

        if(!jacobians){
            return true;
        }
        if(!jacobians[0]){
            return true;
        }

        for(int i = 0; i < 42; ++i)
            jacobians[0][i] = 0.0;

        for(int i = 0; i < 13; ++i) {
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

class ICPResidual : public ceres::SizedCostFunction<1, 6> {
private:
    double* source_points;
    double* target_points;
    double*_source_confidence;
    double* target_confidence;
    int pair_num;
    const double alpha = 100;
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
    ICPResidual(double* source_points, double* target_points, double* source_confidence,
                double* target_confidence, int pair_num){
        this->source_points = source_points;
        this->target_points = target_points;
        this->_source_confidence = source_confidence;
        this->target_confidence = target_confidence;
        this->pair_num = pair_num;
    }

    virtual ~ICPResidual() {}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const { // translation first
        //Vector6d se3Vec;
        //se3Vec << parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5];

        Sophus::SO3 SO3(parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix3d rotation = SO3.matrix();
        Eigen::Vector3d translation;
        translation << parameters[0][0], parameters[0][1], parameters[0][2];

        cout<<rotation<<endl;
        //cout<<translation<<endl;

        double* transformed = new double[pair_num * 3];
        for(int i = 0; i < pair_num; ++i) {
            for (int j = 0; j < 3; ++j) {
                transformed[i * 3 + j] = rotation(j, 0) * source_points[i * 3] +
                                         rotation(j, 1) * source_points[i * 3 + 1] +
                                         rotation(j, 2) * source_points[i * 3 + 2] + translation[j];
            }
        }
        /*for(int i = 0; i < pair_num; ++i) {
            for(int j = 0; j < 3; ++j) {
                cout << transformed[3 * i + j] << '\t';
            }
            cout<<endl;
        }*/

        double* dist = new double[pair_num];
        for(int i = 0; i < pair_num; ++i){
            dist[i] = 0.0;
            for(int j = 0; j < 3; ++j) {
                dist[i] += (transformed[i * 3 + j] - target_points[i * 3 + j]) *
                           (transformed[i * 3 + j] - target_points[i * 3 + j]);
            }
            dist[i] = sqrt(dist[i]) + epsilon;
        }

        /*for(int i = 0; i < pair_num; ++i) {
            cout<<dist[i]<<endl;
        }*/

        residuals[0] = 0.0;
        for(int i = 0; i < pair_num; ++i){
            residuals[0] += alpha * dist[i];
        }

        if(!jacobians){
            delete transformed;
            delete dist;
            return true;
        }
        if(!jacobians[0]){
            delete transformed;
            delete dist;
            return true;
        }

        for(int i = 0; i < 6; ++i) {
            jacobians[0][i] = 0.0;
        }

        //cout<<"SO3 Mat"<<endl;
        for(int i = 0; i < pair_num; ++i) {
            Eigen::Vector3d temp;
            temp << transformed[i * 3], transformed[i * 3 + 1], transformed[i * 3 + 2];
            Eigen::Matrix3d SO3Mat = Sophus::SO3::hat(temp);
            //cout<<SO3Mat<<endl;
            for(int j = 0; j < 3; ++j) {
                //jacobians[0][j] -= (target_points[i * 3 + j] - transformed[i * 3 + j]) / dist[i];
                for(int k = 0; k < 3; ++k) {
                    jacobians[0][3 + k] += SO3Mat(j, k) * (target_points[i * 3 + j] - transformed[i * 3 + j]) / dist[i];
                }
            }
        }

        cout<<"Jacobian"<<endl;
        for(int i = 0; i < 6; ++i) {
            cout<<jacobians[0][i]<<'\t';
        }
        cout<<endl;

        delete transformed;
        delete dist;
        return true;
    }
};


extern "C" void refine_skeleton(void* init_points_, void* guide_points_, void* init_confidence_, void* guide_confidence_,
                                void* bone_length_, void* initial_cost_, void* final_cost_);

extern "C" void guide_transform(void* last_points_, void* guide_points_, void* guide_confidence_,
                                void* init_trans_, void* initial_cost_, void* final_cost_);

extern "C" void ICP_transform(void* source_points_, void* target_points_, void* source_confidence_, void* target_confidence_,
                              void* pair_num_, void* transform_, void* initial_cost_, void* final_cost_);

#endif //SKELETON_OPTIMIZER_OPTIMIZER_H
