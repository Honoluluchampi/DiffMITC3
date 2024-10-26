#include "../src/Mitc3.hpp"

#include "gtest/gtest.h"

template <typename T>
bool vector_eq(const std::vector<T>& a, const std::vector<T>& b)
{
  EXPECT_EQ(a.size(), b.size());
  for (int i = 0; i < a.size(); i++) {
    EXPECT_EQ(a[i], b[i]);
  }

  return true;
}

template <int R, int C>
bool mat_eq(const Matrix<R, C>& a, const Matrix<R, C>& b, double eps = 1.e-6)
{
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      EXPECT_NEAR(a(i, j), b(i, j), eps);
    }
  }
  return true;
}

TEST(math_utils, invdet33)
{
  Matrix<3, 3> a;
  a << 1., 0., 4., 0., 2., 0., 4., 0., 3.;
  // det test
  EXPECT_NEAR(det33(a), -26, 1.e-6);
  // inv test
  Matrix<3,3> aa_inv = inv33(a) * a;
  Matrix<3, 3> b;
  b << 1., 0., 0., 0., 1., 0., 0., 0., 1.;
  mat_eq(aa_inv, b);
}

TEST(math_utils, bsr_math)
{
  // bs_diag_quadratic
  BsArray<Real, 3> diag { std::vector<uint32_t>{ 0, 2 }, 9 };
  std::vector<Real> vec { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  diag.add_block(0, { 0, 1, 2 });
  diag.add_block(2, { 3, 4, 5 });
  std::vector<Real> dense_diag = { 0, 1, 2, 0, 0, 0, 3, 4, 5 };
  EXPECT_EQ(bs_diag_quadratic(diag, vec), 830);
  EXPECT_EQ(diag_quadratic(dense_diag, vec), 830);

  // bs_array * scalar
  auto bs_mat_scalar = 2. * diag;
  const auto& bs_mat_scalar_data = bs_mat_scalar.get_data();
  EXPECT_EQ(bs_mat_scalar_data[0], 0);
  EXPECT_EQ(bs_mat_scalar_data[1], 2);
  EXPECT_EQ(bs_mat_scalar_data[2], 4);
  EXPECT_EQ(bs_mat_scalar_data[3], 6);
  EXPECT_EQ(bs_mat_scalar_data[4], 8);
  EXPECT_EQ(bs_mat_scalar_data[5], 10);

  // bsr_quadratic
  std::vector<std::vector<uint32_t>> adj(3);
  adj[0] = { 0, 2 };
  adj[1] = { 0, 1 };
  adj[2] = { 2 };
  BsrMatrix<3, 3> mat { adj, 3, 3 };
  Matrix<3, 3> block00, block02, block10, block11, block22;
  block00 << 1, 2, 3, 2, 1, 4, 2, 3, 2;
  block02 << 2, 4, 1, 3, 1, 1, 3, 1, 2;
  block10 << 0, 2, 3, 1, 2, 1, 4, 0, 2;
  block11 << 1, 0, 3, 2, 2, 1, 3, 4, 2;
  block22 << 2, 1, 0, 1, 3, 2, 4, 3, 2;
  mat.clear_values();
  mat.add_block(0, 0, block00);
  mat.add_block(0, 2, block02);
  mat.add_block(1, 0, block10);
  mat.add_block(1, 1, block11);
  mat.add_block(2, 2, block22);
  std::vector<Real> dense_vec = { 1, 3, 4, 2, 2, 1, 3, 2, 1 };
  EXPECT_EQ(bsr_quadratic(mat, dense_vec), 441);

  auto mat_diag_sub = bsr_bs_diag_sub(mat, diag);
  const auto& mat_diag_sub_data = mat_diag_sub.get_data();
  EXPECT_EQ(mat_diag_sub_data[0], 1);
  EXPECT_EQ(mat_diag_sub_data[1], 2);
  EXPECT_EQ(mat_diag_sub_data[2], 3);
  EXPECT_EQ(mat_diag_sub_data[3], 2);
  EXPECT_EQ(mat_diag_sub_data[4], 0);
  EXPECT_EQ(mat_diag_sub_data[5], 4);
  EXPECT_EQ(mat_diag_sub_data[6], 2);
  EXPECT_EQ(mat_diag_sub_data[7], 3);
  EXPECT_EQ(mat_diag_sub_data[8], 0);
}

TEST(graph_laplacian, construction)
{
  int num_vtx = 5;
  std::vector<uint32_t> idx_buffer = {
    0, 1, 3, 1, 2, 4, 1, 4, 3
  };

  // answer
  std::vector<uint32_t> indptr  = { 0, 3, 8, 11, 15, 19 };
  std::vector<uint32_t> indices = { 0, 1, 3, 0, 1, 2, 3, 4, 1, 2, 4, 0, 1, 3, 4, 1, 2, 3, 4 };
  std::vector<Real> data = { 2, -1, -1, -1, 4, -1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1, -1, -1, 3 };

  auto laplacian = make_graph_laplacian(idx_buffer, num_vtx);

  vector_eq(laplacian.indptr, indptr);
  vector_eq(laplacian.indices, indices);
  vector_eq(laplacian.data, data);
  // col_offsets test
  EXPECT_EQ(laplacian.col_offsets.size(), 19);
  for (int r = 0; r < num_vtx; r++) {
    for (int c = 0; c < indptr[r + 1] - indptr[r]; c++) {
      EXPECT_EQ(laplacian.col_offsets.at({r, indices[indptr[r] + c]}), c + 1);
    }
  }
}

TEST(mitc3_fem, init)
{
  auto plate = Mitc3::Plate(1., 1., 1., 1., 4, 4, { 0, 1, 2, 1, 3, 2 });
  plate.update_vtx_buffer({ 0., 0., 0.8, 0.2, 0.9, 1., .1, .7 });
  // stiffness test
  plate.calc_stiff_matrix();
  const auto& laplacian = plate.get_graph_laplacian();
  const auto& stiff_bsr = plate.get_stiff_matrix();

  std::vector<uint32_t> indptr = { 0, 3, 7, 11, 14 };
  std::vector<uint32_t> indices = { 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3 };

  vector_eq(indptr, laplacian.indptr);
  vector_eq(indices, laplacian.indices);

  vector_eq(indptr, stiff_bsr.get_indptr());
  vector_eq(indices, stiff_bsr.get_indices());

  plate.calc_mass_diags();
}

TEST(mitc3_fem, area) {
  auto plate = Mitc3::Plate(1., 1., 1., 1., 4, 4, { 0, 1, 2, 1, 3, 2 });
  plate.update_vtx_buffer({ 0., 0., 1., 0., 1., 1., 0., 1. });

  plate.calc_whole_area();
  auto area = plate.get_whole_area();

  EXPECT_NEAR(area, 1., 1e-6);
}