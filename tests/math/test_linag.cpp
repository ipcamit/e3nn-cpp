#include <gtest/gtest.h>
#include "math/_linalg.h"
#include <Eigen/Dense>

namespace e3nn {
namespace math {
namespace test {

TEST(CompleteBasisTest, EmptyInput) {
    Eigen::MatrixXd input(0, 3);
    Eigen::MatrixXd result = complete_basis(input);
    EXPECT_EQ(result.rows(), 3);
    EXPECT_EQ(result.cols(), 3);
    EXPECT_TRUE((result * result.transpose() - Eigen::MatrixXd::Identity(result.rows(), result.rows())).norm() < 1e-9);
}

TEST(CompleteBasisTest, SingleVector) {
    Eigen::MatrixXd input(1, 3);
    input << 1, 0, 0;
    Eigen::MatrixXd result = complete_basis(input);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 3);
    EXPECT_TRUE((result * result.transpose() - Eigen::MatrixXd::Identity(result.rows(), result.rows())).norm() < 1e-9);
}

TEST(CompleteBasisTest, TwoOrthogonalVectors) {
    Eigen::MatrixXd input(2, 3);
    input << 1, 0, 0,
             0, 1, 0;
    Eigen::MatrixXd result = complete_basis(input);
    EXPECT_EQ(result.rows(), 1);
    EXPECT_EQ(result.cols(), 3);
    EXPECT_TRUE(result.isApprox(Eigen::MatrixXd::Zero(1, 3) + Eigen::Vector3d(0, 0, 1).transpose(), 1e-9));
}

TEST(CompleteBasisTest, ThreeOrthogonalVectors) {
    Eigen::MatrixXd input(3, 3);
    input << 1, 0, 0,
             0, 1, 0,
             0, 0, 1;
    Eigen::MatrixXd result = complete_basis(input);
    EXPECT_EQ(result.rows(), 0);
    EXPECT_EQ(result.cols(), 3);
}

TEST(CompleteBasisTest, NonOrthogonalVectors) {
    Eigen::MatrixXd input(2, 3);
    input << 1, 1, 0,
             1, 0, 1;
    Eigen::MatrixXd result = complete_basis(input);
    EXPECT_EQ(result.rows(), 2);
    EXPECT_EQ(result.cols(), 3);
    EXPECT_TRUE((result * result.transpose() - Eigen::MatrixXd::Identity(result.rows(), result.rows())).norm() < 1e-9);

    // This test is tricky, it fails for single precision
    // Testing exact match
    Eigen::MatrixXd torch_output(2, 3);
    torch_output << 0.408248290463863, -0.816496580927726, -0.408248290463863,
                    0.577350269189625,  0.577350269189626, -0.577350269189626;
    EXPECT_TRUE((result - torch_output).norm() < 1e-9);
}

}  // namespace test
}  // namespace math
}  // namespace e3nn