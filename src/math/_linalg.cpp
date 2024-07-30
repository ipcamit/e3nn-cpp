#include <Eigen/Dense>
#include <vector>
#include "math/_linalg.h"

namespace e3nn {
namespace math {


    Eigen::MatrixXd complete_basis(const Eigen::MatrixXd &vecs, double eps) {
        assert(vecs.cols() > 0 && "Input matrix must have at least one column");

        int dim = vecs.cols();
        std::vector<Eigen::VectorXd> base;

        // Normalize input vectors
        for (int i = 0; i < vecs.rows(); ++i) {
            Eigen::VectorXd v = vecs.row(i);
            base.push_back(v.normalized());
        }

        std::vector<Eigen::VectorXd> expand;

        for (int i = 0; i < dim; ++i) {
            Eigen::VectorXd x = Eigen::VectorXd::Unit(dim, i);

            for (const auto &y: base) {
                x -= x.dot(y) * y;
            }
            for (const auto &y: expand) {
                x -= x.dot(y) * y;
            }

            if (x.norm() > 2 * eps) {
                x.normalize();
                x = (x.array().abs() < eps).select(0, x);

                int first_non_zero = 0;
                while (first_non_zero < dim && std::abs(x(first_non_zero)) < eps) {
                    ++first_non_zero;
                }

                if (first_non_zero < dim) {
                    x *= (x(first_non_zero) > 0) ? 1 : -1;
                }

                expand.push_back(x);
            }
        }

        Eigen::MatrixXd result;
        if (!expand.empty()) {
            result.resize(expand.size(), dim);
            for (size_t i = 0; i < expand.size(); ++i) {
                result.row(i) = expand[i];
            }
        } else {
            result.resize(0, dim);
        }

        return result;
    }

}  // namespace math
}  // namespace e3nn