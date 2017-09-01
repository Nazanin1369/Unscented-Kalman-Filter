#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2; //30 ! Widely off

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;  //30 ! Widely off

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // Time when the state is true, in us
  time_us_ = 0.0;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_ + 1);

  // NIS for radar
  NIS_radar_ = 0.0;

  // NIS for laser
  NIS_laser_ = 0.0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
    if ((use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) ||
        (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)) {
        if (!is_initialized_) {
            Initialize(meas_package);

            return;
        }

        float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;

        time_us_ = meas_package.timestamp_;

        Prediction(dt, meas_package);

        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            UpdateLidar(meas_package);
        } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            UpdateRadar(meas_package);
        }
    }
}

void UKF::PredictRadarMeasMeanCov(VectorXd *z_pred_out, MatrixXd *S_out, MatrixXd *Zsig_out)
{
    int n_z = 3;
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v  = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);
        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;
        double px_2 = p_x * p_x;
        double py_2 = p_y * p_y;

        Zsig(0, i) = sqrt(px_2 + py_2);
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x * v1 + p_y * v2 ) / sqrt(px_2 + py_2);
    }

    //calculate mean predicted measurement
    z_pred.fill(0.0); // ?

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //calculate measurement covariance matrix S
    S.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        while (z_diff(1) > M_PI) {
            z_diff(1) -= 2. * M_PI;
        }

        while (z_diff(1) < -M_PI) {
            z_diff(1) += 2. * M_PI;
        }

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);

    R << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0,std_radrd_ * std_radrd_;
    S = S + R;

    *z_pred_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

void UKF::PredictLidarMeasMeanCov(VectorXd *z_pred_out, MatrixXd *S_out, MatrixXd *Zsig_out)
{
    int n_z = 2;
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //calculate mean predicted measurement
    z_pred.fill(0.0); // ?

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

    //calculate measurement covariance matrix S
    S.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    MatrixXd R = MatrixXd(n_z, n_z);

    R << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;
    S = S + R;

    *z_pred_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t, MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
    MatrixXd Xsig = GenerateSigmaPoints();
    MatrixXd Xsig_aug = AugmentSigmaPoints(Xsig);

    PredictSigmaPoints(Xsig_aug, delta_t);
    PredictStateMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
    int n_z = 2;

    VectorXd z = meas_package.raw_measurements_;
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    PredictLidarMeasMeanCov(&z_pred, &S, &Zsig);

    //calculate cross correlation matrix
    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = z - z_pred;

    //calculate NIS
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
    int n_z = 3;

    VectorXd z = meas_package.raw_measurements_;
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    VectorXd z_pred = VectorXd(n_z);
    MatrixXd S = MatrixXd(n_z, n_z);
    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

    PredictRadarMeasMeanCov(&z_pred, &S, &Zsig);

    //calculate cross correlation matrix
    Tc.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        while (z_diff(1) > M_PI) {
            z_diff(1) -= 2. * M_PI;
        }

        while (z_diff(1) < -M_PI) {
            z_diff(1) += 2. * M_PI;
        }

        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        while (x_diff(3) > M_PI) {
            x_diff(3) -= 2. * M_PI;
        }

        while (x_diff(3) < -M_PI) {
            x_diff(3) += 2. * M_PI;
        }

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //calculate Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    VectorXd z_diff = z - z_pred;

    while (z_diff(1) > M_PI) {
        z_diff(1) -= 2. * M_PI;
    }

    while (z_diff(1) < -M_PI) {
        z_diff(1) += 2. * M_PI;
    }

    //calculate NIS
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();
}
