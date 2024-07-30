#pragma once

#include <Eigen/Dense>

namespace e3nn {
namespace math {

Eigen::MatrixXd complete_basis(const Eigen::MatrixXd& vecs, double eps = 1e-9);

}  // namespace math
}  // namespace e3nn