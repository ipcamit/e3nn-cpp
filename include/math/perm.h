#pragma once

#include <vector>
#include <set>
#include <random>
#include <Eigen/Dense>

namespace e3nn {
namespace math {

using Perm = std::vector<int>;
using PermSet = std::set<Perm>;

bool is_perm(const Perm& p);
Perm identity(int n);
Perm compose(const Perm& p1, const Perm& p2);
Perm inverse(const Perm& p);
Perm rand(int n);
Perm from_int(int i, int n);
int to_int(const Perm& p);
PermSet group(int n);
PermSet germinate(const PermSet& subset);
bool is_group(const PermSet& g);
std::set<Perm> to_cycles(const Perm& p);
int sign(const Perm& p);
Eigen::MatrixXd standard_representation(const Perm& p);
Eigen::MatrixXd natural_representation(const Perm& p);

} // namespace math
} // namespace e3nn
