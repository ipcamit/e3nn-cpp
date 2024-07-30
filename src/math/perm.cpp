#include "math/perm.h"
#include "math/_linalg.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace e3nn {
namespace math {

bool is_perm(const Perm& p) {
    Perm sorted_p = p;
    std::sort(sorted_p.begin(), sorted_p.end());
    std::vector<int> range(p.size());
    std::iota(range.begin(), range.end(), 0);
    return std::equal(sorted_p.begin(), sorted_p.end(), range.begin());
}

Perm identity(int n) {
    Perm result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

Perm compose(const Perm& p1, const Perm& p2) {
    if (!is_perm(p1) || !is_perm(p2) || p1.size() != p2.size()) {
        throw std::invalid_argument("Invalid permutations for composition");
    }
    Perm result(p1.size());
    for (size_t i = 0; i < p1.size(); ++i) {
        result[i] = p1[p2[i]];
    }
    return result;
}

Perm inverse(const Perm& p) {
    Perm result(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        result[p[i]] = i;
    }
    return result;
}

Perm rand(int n) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, std::tgamma(n + 1) - 1);
    return from_int(dis(gen), n);
}

Perm from_int(int i, int n) {
    std::vector<int> pool(n);
    std::iota(pool.begin(), pool.end(), 0);
    Perm p;
    const int range = n;
    for (int _ = 0; _ < range; ++_) {
        int j = i % n;
        i /= n;
        p.push_back(pool[j]);
        pool.erase(pool.begin() + j);
        --n;
    }
    return p;
}

int to_int(const Perm& p) {
    int n = p.size();
    std::vector<int> pool(n);
    std::iota(pool.begin(), pool.end(), 0);
    int i = 0;
    int m = 1;
    for (int j : p) {
        auto it = std::find(pool.begin(), pool.end(), j);
        int k = std::distance(pool.begin(), it);
        i += k * m;
        m *= pool.size();
        pool.erase(it);
    }
    return i;
}

PermSet group(int n) {
    PermSet result;
    for (int i = 0; i < std::tgamma(n + 1); ++i) {
        result.insert(from_int(i, n));
    }
    return result;
}

PermSet germinate(const PermSet& subset) {
    PermSet result = subset;
    while (true) {
        size_t n = result.size();
        for (const auto& p : subset) {
            result.insert(inverse(p));
        }
        for (const auto& p1 : result) {
            for (const auto& p2 : result) {
                result.insert(compose(p1, p2));
            }
        }
        if (result.size() == n) {
            return result;
        }
    }
}

bool is_group(const PermSet& g) {
    if (g.empty()) return false;
    int n = g.begin()->size();
    for (const auto& p : g) {
        if (p.size() != n) return false;
    }
    if (g.find(identity(n)) == g.end()) return false;
    for (const auto& p : g) {
        if (g.find(inverse(p)) == g.end()) return false;
    }
    for (const auto& p1 : g) {
        for (const auto& p2 : g) {
            if (g.find(compose(p1, p2)) == g.end()) return false;
        }
    }
    return true;
}

std::set<Perm> to_cycles(const Perm& p) {
    std::set<Perm> cycles;
    std::vector<bool> visited(p.size(), false);
    for (size_t i = 0; i < p.size(); ++i) {
        if (!visited[i]) {
            Perm cycle;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                cycle.push_back(j);
                j = p[j];
            }
            if (cycle.size() >= 2) {
                auto min_it = std::min_element(cycle.begin(), cycle.end());
                std::rotate(cycle.begin(), min_it, cycle.end());
                cycles.insert(cycle);
            }
        }
    }
    return cycles;
}

int sign(const Perm& p) {
    int s = 1;
    for (const auto& c : to_cycles(p)) {
        if (c.size() % 2 == 0) {
            s = -s;
        }
    }
    return s;
}

Eigen::MatrixXd standard_representation(const Perm& p) {
    int n = p.size();
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(1, n);
    Eigen::MatrixXd A = complete_basis(ones, 0.1 / n);
    return A * natural_representation(p) * A.transpose();
}

Eigen::MatrixXd natural_representation(const Perm& p) {
    int n = p.size();
    Perm ip = inverse(p);
    Eigen::MatrixXd d = Eigen::MatrixXd::Zero(n, n);
    for (int a = 0; a < n; ++a) {
        d(a, ip[a]) = 1;
    }
    return d;
}

} // namespace math
} // namespace e3nn
