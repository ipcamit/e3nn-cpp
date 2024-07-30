#include <gtest/gtest.h>
#include "math/perm.h"

namespace e3nn {
namespace math {
namespace test {

TEST(PermTest, IsPerm) {
    EXPECT_TRUE(is_perm({0, 1, 2, 3}));
    EXPECT_TRUE(is_perm({3, 2, 1, 0}));
    EXPECT_FALSE(is_perm({0, 1, 1, 3}));
    EXPECT_FALSE(is_perm({0, 1, 2, 4}));
}

TEST(PermTest, Identity) {
    EXPECT_EQ(identity(4), Perm({0, 1, 2, 3}));
    EXPECT_EQ(identity(1), Perm({0}));
}

TEST(PermTest, Compose) {
    EXPECT_EQ(compose({1, 2, 0}, {2, 0, 1}), Perm({0, 1, 2}));
    EXPECT_EQ(compose({0, 1, 2}, {2, 1, 0}), Perm({2, 1, 0}));
}

TEST(PermTest, Inverse) {
    EXPECT_EQ(inverse({1, 2, 0}), Perm({2, 0, 1}));
    EXPECT_EQ(inverse({3, 1, 0, 2}), Perm({2, 1, 3, 0}));
}

TEST(PermTest, FromInt) {
    EXPECT_EQ(from_int(0, 4), Perm({0, 1, 2, 3}));
    EXPECT_EQ(from_int(1, 4), Perm({1, 0, 2, 3}));
    EXPECT_EQ(from_int(9, 4), Perm({1, 3, 0, 2}));
}

TEST(PermTest, FromIntToInt) {
    for (int i = 0; i < 24; ++i) {  // 4! = 24
        EXPECT_EQ(to_int(from_int(i, 4)), i);
    }
}

TEST(PermTest, Group) {
    auto g = group(3);
    EXPECT_EQ(g.size(), 6);  // 3! = 6
    EXPECT_TRUE(is_group(g));
}

TEST(PermTest, GerminateTestGroup) {
    PermSet subset = {{0, 1, 2}, {1, 0, 2}};
    auto g = germinate(subset);
    EXPECT_EQ(g.size(), 2);
    EXPECT_TRUE(is_group(g));
}

TEST(PermTest, GerminateTestIncompleteGroup) {
    PermSet subset = {{0, 2, 1}, {1, 0, 2}};
    auto g = germinate(subset);
    EXPECT_EQ(g.size(), 6);  // Should generate full S3 group
    EXPECT_TRUE(is_group(g));
    // {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}
    EXPECT_TRUE(g.find({0, 1, 2}) != g.end());
    EXPECT_TRUE(g.find({0, 2, 1}) != g.end());
    EXPECT_TRUE(g.find({1, 0, 2}) != g.end());
    EXPECT_TRUE(g.find({1, 2, 0}) != g.end());
    EXPECT_TRUE(g.find({2, 0, 1}) != g.end());
    EXPECT_TRUE(g.find({2, 1, 0}) != g.end());
}

TEST(PermTest, ToCycles) {
    auto cycles = to_cycles({1, 2, 0, 4, 3});
    EXPECT_EQ(cycles.size(), 2);
    EXPECT_TRUE(cycles.find({0, 1, 2}) != cycles.end());
    EXPECT_TRUE(cycles.find({3, 4}) != cycles.end());
}

TEST(PermTest, Sign) {
    EXPECT_EQ(sign({0, 1, 2}), 1);
    EXPECT_EQ(sign({1, 0, 2}), -1);
    EXPECT_EQ(sign({1, 2, 3, 0}), -1);
    EXPECT_EQ(sign({0, 2, 3, 1}), 1);
}

TEST(PermTest, NaturalRepresentation) {
    Eigen::MatrixXd expected(3, 3);
    expected << 0, 0, 1,
                1, 0, 0,
                0, 1, 0;
    EXPECT_TRUE(natural_representation({1, 2, 0}).isApprox(expected));
}

TEST(PermTest, StandardRepresentation) {
    Perm p = {1, 2, 0};
    auto rep = standard_representation(p);
    EXPECT_EQ(rep.rows(), 2);
    EXPECT_EQ(rep.cols(), 2);
    // You might want to add more specific checks for the matrix values
}

TEST(PermTest, RandPerm) {
    auto p = rand(4);
    EXPECT_EQ(p.size(), 4);
    EXPECT_TRUE(is_perm(p));
}

}  // namespace test
}  // namespace math
}  // namespace e3nn
