#pragma once

#include "math_util.hpp"

#include <vector>
#include <memory>

namespace Mitc3 {

using Id_t = uint32_t;
using Vec2 = Eigen::Vector2<Real>;
using Vec3 = Eigen::Vector3<Real>;

Real calc_area(Real x0, Real y0, Real x1, Real y1, Real x2, Real y2)
{ return std::abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)) / 2.; }

std::array<Real, 6> calc_darea(Real x0, Real y0, Real x1, Real y1, Real x2, Real y2)
{
  int sign = 2 * int(calc_area(x0, y0, x1, y1, x2, y2) > 0) - 1;
  // 2 * area = x1*y2 - x1*y0 - x0*y2 + x0*y0 - y1*x2 + y1*x0 + y0*x2 - y0*x0;
  return {
    Real(sign) * (-y2 + y1) / Real(2.), // dx0
    Real(sign) * (-x1 + x2) / Real(2.), // dy0
    Real(sign) * (y2 - y0) / Real(2.), // dx1
    Real(sign) * (-x2 + x0) / Real(2.), // dy1
    Real(sign) * (-y1 + y0) / Real(2.), // dx2
    Real(sign) * (x1 - x0) / Real(2.)  // dy2
  };
}

Vec3 contravariant_basis_derivative(const Vec3& g_m, const Vec3& g_n, const Vec3& g_o, const Vec3& dg_m, const Vec3& dg_n, const Vec3& dg_o)
{
  Real triplet = g_m.dot(g_n.cross(g_o));
  Vec3 g_no = g_n.cross(g_o);
  Vec3 dg_no = dg_n.cross(g_o) + g_n.cross(dg_o);
  Vec3 ret = (triplet * dg_no - (dg_m.dot(g_no) + g_m.dot(dg_no)) * g_no) / (triplet * triplet);

  return ret;
}

class Plate
{
  public:
    explicit Plate(
          const Real thick_,
          const Real lambda_,
          const Real myu_,
          const Real rho_,
          const int num_vtx_,
          const int num_edge_vtx_,
          const std::vector<Id_t>& idx_buffer_) // v00,v01,v02,v10,v11,v12,...
          : thick(thick_), lambda(lambda_), myu(myu_), rho(rho_)
    {
      num_vtx = num_vtx_;
      num_edge_vtx = num_edge_vtx_;
      num_triangle = idx_buffer_.size() / 3;

      // copy
      idx_buffer = idx_buffer_;

      // share the adjacency information with stiffness matrix
      graph_laplacian = make_graph_laplacian(idx_buffer, num_vtx);

      stiff_matrix = std::make_unique<BsrMatrix<3, 3>>(
        graph_laplacian.indptr,
        graph_laplacian.indices,
        graph_laplacian.col_offsets,
        num_vtx, num_vtx);

      mass_diags.resize(num_vtx * 3, 0.);

      // derivatives
      stiff_derivatives.resize(num_edge_vtx * 2);
      for (int i = 0; i < num_edge_vtx * 2; i++) {
        stiff_derivatives[i] = std::make_unique<BsrMatrix<3, 3>>(
          make_symmetric_bsr<3, 3>(
            i / 2,
            graph_laplacian.indptr,
            graph_laplacian.indices,
            graph_laplacian.col_offsets
        ));
      }

      // mass derivative
      mass_derivatives.resize(num_edge_vtx * 2);
      for (int i = 0; i < num_edge_vtx * 2; i++) {
        mass_derivatives[i] = std::make_unique<BsArray<Real, 3>>(
          graph_laplacian.indices,
          graph_laplacian.indptr[i / 2],
          graph_laplacian.indptr[i / 2 + 1], num_vtx * 3
        );
      }

      // for eigen value derivative
      // use full-size vector to make it easy to be combined with pytorch
      eig_val_derivatives.resize(num_vtx * 2, 0.);
    }

    // stiff matrix and its derivatives w.r.t X0, Y0, ..., Y2
    // TODO : reduce derivative calculation
    std::array<Eigen::Matrix<Real, 9, 9>, 7> elem_stiff_matrix(Id_t n0, Id_t n1, Id_t n2)
    {
      auto zero_mat = Eigen::Matrix<Real, 9, 9>::Zero();
      std::array<Eigen::Matrix<Real, 9, 9>, 7> ret =  { zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat, zero_mat };

      auto x0 = vtx_buffer[2 * n0 + 0];
      auto y0 = vtx_buffer[2 * n0 + 1];
      auto x1 = vtx_buffer[2 * n1 + 0];
      auto y1 = vtx_buffer[2 * n1 + 1];
      auto x2 = vtx_buffer[2 * n2 + 0];
      auto y2 = vtx_buffer[2 * n2 + 1];

      auto tri_area = calc_area(x0, y0, x1, y1, x2, y2);
      auto dtri_area = calc_darea(x0, y0, x1, y1, x2, y2);

      auto one = Real(1.);
      auto two = Real(2.);
      auto three = Real(3.);
      auto four = Real(4.);

      // covariant basis G_{r,s,t}
      Vec3 g_r[3] = {
        { x1 - x0, y1 - y0, 0 },
        { x2 - x0, y2 - y0, 0 },
        { 0, 0, thick / two }
      };

      // derivative of G_{r,s,t} w.r.t. vtx coord.
      Vec3 dg_r[3][2][3] = { // v_id, XY, r,s,t
        { // v0
          { { -1, 0, 0 }, { -1, 0, 0 }, { 0, 0, 0 } }, // x0
          { { 0, -1, 0 }, { 0, -1, 0 }, { 0, 0, 0 } }, // y0
        },
        { // v1
          { { 1, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }, // x1
          { { 0, 1, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }, // y1
        },
        { // v2
          { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 } }, // x2
          { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } }, // y2
        }
      };

      // contravariant basis G^{r,s,t}
      Vec3 gr[3];
      Real triplet = g_r[0].dot(g_r[1].cross(g_r[2]));
      gr[0] = g_r[1].cross(g_r[2]) / triplet;
      gr[1] = g_r[2].cross(g_r[0]) / triplet;
      gr[2] = g_r[0].cross(g_r[1]) / triplet;

      // derivative of G^{r,s,t} w.r.t. vtx coord
      Vec3 dgr[3][2][3];
      for (int k = 0; k < 3; k++) {
        for (int xy = 0; xy < 2; xy++) {
          dgr[k][xy][0] = contravariant_basis_derivative(g_r[0], g_r[1], g_r[2], dg_r[k][xy][0], dg_r[k][xy][1], dg_r[k][xy][2]);
          dgr[k][xy][1] = contravariant_basis_derivative(g_r[1], g_r[2], g_r[0], dg_r[k][xy][1], dg_r[k][xy][2], dg_r[k][xy][0]);
          dgr[k][xy][2] = contravariant_basis_derivative(g_r[2], g_r[0], g_r[1], dg_r[k][xy][2], dg_r[k][xy][0], dg_r[k][xy][1]);
        }
      }

      // inner product of contravariant basis G^{rr, ss, rs, tt}
      // note: G^rt == G^st == 0
      Real grr[4] = {
        gr[0].dot(gr[0]), // rr
        gr[1].dot(gr[1]), // ss
        gr[0].dot(gr[1]), // rs
        gr[2].dot(gr[2])  // tt
      };

      Real dgrr[3][2][4];
      for (int k = 0; k < 3; k++) {
        for (int xy = 0; xy < 2; xy++) {
          dgrr[k][xy][0] = dgr[k][xy][0].dot(gr[0]) * two; // rr
          dgrr[k][xy][1] = dgr[k][xy][1].dot(gr[1]) * two; // ss
          dgrr[k][xy][2] = dgr[k][xy][0].dot(gr[1]) + gr[0].dot(dgr[k][xy][1]); // rs
          dgrr[k][xy][3] = dgr[k][xy][2].dot(gr[2]) * two; // tt
        }
      }

      // integration for rr, ss, rs -------------------------------------
      {
        // constitution matrix for rr, ss, rs
        Real c[3][3] = {
          {
            lambda * grr[0] * grr[0] + two * myu * grr[0] * grr[0],
            lambda * grr[0] * grr[1] + two * myu * grr[2] * grr[2],
            (lambda * grr[0] * grr[2] + two * myu * grr[0] * grr[2]) * two
          },
          {
            lambda * grr[1] * grr[0] + two * myu * grr[2] * grr[2],
            lambda * grr[1] * grr[1] + two * myu * grr[1] * grr[1],
            (lambda * grr[1] * grr[2] + two * myu * grr[1] * grr[2]) * two
          },
          {
            (lambda * grr[2] * grr[0] + two * myu * grr[2] * grr[0]) * two,
            (lambda * grr[2] * grr[1] + two * myu * grr[2] * grr[1]) * two,
            (lambda * grr[2] * grr[2] + myu * (grr[0] * grr[1] + grr[2] * grr[2])) * four
          }
        };

        Real dc[3][2][3][3];
        for (int k = 0; k < 3; k++) {
          for (int xy = 0; xy < 2; xy++) {
            dc[k][xy][0][0] =  lambda * (dgrr[k][xy][0] * grr[0] + grr[0] * dgrr[k][xy][0]) + two * myu * (dgrr[k][xy][0] * grr[0] + grr[0] * dgrr[k][xy][0]);
            dc[k][xy][0][1] =  lambda * (dgrr[k][xy][0] * grr[1] + grr[0] * dgrr[k][xy][1]) + two * myu * (dgrr[k][xy][2] * grr[2] + grr[2] * dgrr[k][xy][2]);
            dc[k][xy][0][2] = (lambda * (dgrr[k][xy][0] * grr[2] + grr[0] * dgrr[k][xy][2]) + two * myu * (dgrr[k][xy][0] * grr[2] + grr[0] * dgrr[k][xy][2])) * two;
            dc[k][xy][1][0] =  lambda * (dgrr[k][xy][1] * grr[0] + grr[1] * dgrr[k][xy][0]) + two * myu * (dgrr[k][xy][2] * grr[2] + grr[2] * dgrr[k][xy][2]);
            dc[k][xy][1][1] =  lambda * (dgrr[k][xy][1] * grr[1] + grr[1] * dgrr[k][xy][1]) + two * myu * (dgrr[k][xy][1] * grr[1] + grr[1] * dgrr[k][xy][1]);
            dc[k][xy][1][2] = (lambda * (dgrr[k][xy][1] * grr[2] + grr[1] * dgrr[k][xy][2]) + two * myu * (dgrr[k][xy][1] * grr[2] + grr[1] * dgrr[k][xy][2])) * two;
            dc[k][xy][2][0] = (lambda * (dgrr[k][xy][2] * grr[0] + grr[2] * dgrr[k][xy][0]) + two * myu * (dgrr[k][xy][2] * grr[0] + grr[2] * dgrr[k][xy][0])) * two;
            dc[k][xy][2][1] = (lambda * (dgrr[k][xy][2] * grr[1] + grr[2] * dgrr[k][xy][1]) + two * myu * (dgrr[k][xy][2] * grr[1] + grr[2] * dgrr[k][xy][1])) * two;
            dc[k][xy][2][2] = (lambda * (dgrr[k][xy][2] * grr[2] + grr[2] * dgrr[k][xy][2]) + myu * (dgrr[k][xy][0] * grr[1] + grr[0] * dgrr[k][xy][1] + dgrr[k][xy][2] * grr[2] + grr[2] * dgrr[k][xy][2])) * four;
          }
        }

        // all the weights are 1
        Real integration_t[2] = { -one / Real(sqrt(three)), one / Real(sqrt(three)) };

        for (const auto& t : integration_t) {
          // derivative of displacement
          Vec3 dudrda[3][3][2]; // v_id, dof, natural coord(r, s)
          // alpha0
          dudrda[0][0][0] = { 0, thick * t / two, 0 };
          dudrda[0][0][1] = { 0, thick * t / two, 0 };
          // beta0
          dudrda[0][1][0] = { -thick * t / two, 0, 0 };
          dudrda[0][1][1] = { -thick * t / two, 0, 0 };
          // u_z0
          dudrda[0][2][0] = { 0, 0, -one };
          dudrda[0][2][1] = { 0, 0, -one };
          // alpha1
          dudrda[1][0][0] = { 0, -thick * t / two, 0 };
          dudrda[1][0][1] = { 0, 0, 0 };
          // beta1
          dudrda[1][1][0] = { thick * t / two, 0, 0 };
          dudrda[1][1][1] = { 0, 0, 0 };
          // u_z1
          dudrda[1][2][0] = { 0, 0, one };
          dudrda[1][2][1] = { 0, 0, 0 };
          // alpha2
          dudrda[2][0][0] = { 0, 0, 0 };
          dudrda[2][0][1] = { 0, -thick * t / two, 0 };
          // beta2
          dudrda[2][1][0] = { 0, 0, 0 };
          dudrda[2][1][1] = { thick * t / two, 0, 0 };
          // u_z2
          dudrda[2][2][0] = { 0, 0, 0 };
          dudrda[2][2][1] = { 0, 0, one };

          Real deda[3][3][3]; // v_id, dof, rr-ss-rs
          for (int i = 0; i < 3; i++)  { // v_id
            for (int a = 0; a < 3; a++) { // dof
              // rr
              deda[i][a][0] = g_r[0].dot(dudrda[i][a][0]);
              // ss
              deda[i][a][1] = g_r[1].dot(dudrda[i][a][1]);
              // rs
              deda[i][a][2] = (g_r[0].dot(dudrda[i][a][1]) + g_r[1].dot(dudrda[i][a][0])) / two;
            }
          }

          // sum up stiff matrix
          for (int i = 0; i < 3; i++) { // v_id
            for (int j = 0; j < 3; j++) { // v_id
              for (int a = 0; a < 3; a++) { // dof
                for (int b = 0; b < 3; b++) { // dof
                  Real tmp = 0.;
                  for (int m = 0; m < 3; m++) { // natural coord (rr, ss, rs).
                    for (int n = 0; n < 3; n++) { // natural coord (rr, ss, rs).
                      tmp += c[m][n] * deda[i][a][m] * deda[j][b][n];
                    }
                  }
                  ret[0](3 * i + a, 3 * j + b) += tri_area * thick / two * tmp;
                }
              }
            }
          }

          // sum up derivatives
          for (int k = 0; k < 3; k++) { // v_id
            for (int xy = 0; xy < 2; xy++) { // xy==0 -> X, xy==1 -> Y
              Real dedadx[3][3][3]; // v_id, dof, rr-ss-rs
              for (int i = 0; i < 3; i++)  { // v_id
                for (int a = 0; a < 3; a++) { // dof
                  // rr
                  dedadx[i][a][0] = dg_r[k][xy][0].dot(dudrda[i][a][0]);
                  // ss
                  dedadx[i][a][1] = dg_r[k][xy][1].dot(dudrda[i][a][1]);
                  // rs
                  dedadx[i][a][2] = (dg_r[k][xy][0].dot(dudrda[i][a][1]) + dg_r[k][xy][1].dot(dudrda[i][a][0])) / two;
                }
              }

              for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                  for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                      Real tmp = 0;
                      Real dtmp = 0;
                      for (int m = 0; m < 3; m++) {
                        for (int n = 0; n < 3; n++) {
                          tmp += c[m][n] * deda[i][a][m] * deda[j][b][n];
                          dtmp += (dc[k][xy][m][n] * deda[i][a][m] * deda[j][b][n]) + (c[m][n] * dedadx[i][a][m] * deda[j][b][n]) + (c[m][n] * deda[i][a][m] * dedadx[j][b][n]);
                        }
                      }
                      ret[2 * k + xy + 1](3 * i + a, 3 * j + b) += (dtri_area[2 * k + xy] * thick / two * tmp) + (tri_area * thick / two * dtmp);
                    }
                  }
                }
              }
            }
          } // sum up derivatives
        } // integration loop
      } // integration for rt st

      // integration for rt, st
      {
        // constitution matrix for rt, st
        Real c[2][2] = {
          {
            (myu * grr[0] * grr[3]) * four,
            (myu * grr[2] * grr[3]) * four
          },
          {
            (myu * grr[2] * grr[3]) * four,
            (myu * grr[1] * grr[3]) * four
          }
        };

        Real dc[3][2][2][2];
        for (int k = 0; k < 3; k++) {
          for (int xy = 0; xy < 2; xy++) {
            dc[k][xy][0][0] = myu * (dgrr[k][xy][0] * grr[3] + grr[0] * dgrr[k][xy][3]) * four;
            auto elem01 = myu * (dgrr[k][xy][2] * grr[3] + grr[2] * dgrr[k][xy][3]) * four;
            dc[k][xy][0][1] = elem01;
            dc[k][xy][1][0] = elem01;
            dc[k][xy][1][1] = myu * (dgrr[k][xy][1] * grr[3] + grr[1] * dgrr[k][xy][3]) * four;
          }
        }

        // trying points for assumed strain
        Real abc_coord[3][2] = {
          { one / two, 0 }, // a
          { 0, one / two }, // b
          { one / two, one /two } // c
        };

        Vec3 trying_dudrda[3][3][3][3]; // abc, v_id, dof, rst
        for (int abc_loop = 0; abc_loop < 3; abc_loop++) {
          Real r_ = abc_coord[abc_loop][0];
          Real s_ = abc_coord[abc_loop][1];
          Real t_ = 0; // dummy t

          int alpha = 0, beta = 1, u_z = 2;
          // alpha0
          trying_dudrda[abc_loop][0][alpha][0] = { 0, thick * t_ / two, 0 };
          trying_dudrda[abc_loop][0][alpha][1] = { 0, thick * t_ / two, 0 };
          trying_dudrda[abc_loop][0][alpha][2] = { 0, -thick / two * (one - r_ - s_), 0 };
          // beta0
          trying_dudrda[abc_loop][0][beta][0] = { -thick * t_ / two, 0, 0 };
          trying_dudrda[abc_loop][0][beta][1] = { -thick * t_ / two, 0, 0 };
          trying_dudrda[abc_loop][0][beta][2] = { thick / two * (one - r_ - s_), 0, 0 };
          // u_z0
          trying_dudrda[abc_loop][0][u_z][0] = { 0, 0, -one };
          trying_dudrda[abc_loop][0][u_z][1] = { 0, 0, -one };
          trying_dudrda[abc_loop][0][u_z][2] = { 0, 0, 0 };
          // alpha1
          trying_dudrda[abc_loop][1][alpha][0] = { 0, -thick * t_ / two, 0 };
          trying_dudrda[abc_loop][1][alpha][1] = { 0, 0, 0 };
          trying_dudrda[abc_loop][1][alpha][2] = { 0, -thick / two * r_, 0 };
          // beta1
          trying_dudrda[abc_loop][1][beta][0] = { thick * t_ / two, 0, 0 };
          trying_dudrda[abc_loop][1][beta][1] = { 0, 0, 0};
          trying_dudrda[abc_loop][1][beta][2] = { thick / two * r_, 0, 0 };
          // u_z1
          trying_dudrda[abc_loop][1][u_z][0] = { 0, 0, one };
          trying_dudrda[abc_loop][1][u_z][1] = { 0, 0, 0 };
          trying_dudrda[abc_loop][1][u_z][2] = { 0, 0, 0 };
          // alpha2
          trying_dudrda[abc_loop][2][alpha][0] = { 0, 0, 0 };
          trying_dudrda[abc_loop][2][alpha][1] = { 0, -thick * t_ / two, 0 };
          trying_dudrda[abc_loop][2][alpha][2] = { 0, -thick / two * s_, 0 };
          // beta2
          trying_dudrda[abc_loop][2][beta][0] = { 0, 0, 0 };
          trying_dudrda[abc_loop][2][beta][1] = { thick * t_ / two, 0, 0 };
          trying_dudrda[abc_loop][2][beta][2] = { thick / two * s_, 0, 0 };
          // u_z2
          trying_dudrda[abc_loop][2][u_z][0] = { 0, 0, 0 };
          trying_dudrda[abc_loop][2][u_z][1] = { 0, 0, one };
          trying_dudrda[abc_loop][2][u_z][2] = { 0, 0, 0 };
        } // abc_loop

        // all the weights is 1/3
        Real integration_r[3] = { one / two, one / two, 0 };
        Real integration_s[3] = { 0, one / two, one / two };
        Real weight = one / three;

        for (int integration_i = 0; integration_i < 3; integration_i++) {
          auto r = integration_r[integration_i];
          auto s = integration_s[integration_i];

          // derivative of the "assumed" strain
          Real deda[3][3][2]; // v_id, dof, (rt, st)
          for (int i = 0; i < 3; i++)  { // v_id
            for (int a = 0; a < 3; a++) { // dof
              // rt
              Real deda_rt_A = (g_r[0].dot(trying_dudrda[0][i][a][2]) + g_r[2].dot(trying_dudrda[0][i][a][0])) / two;
              Real deda_rt_C = (g_r[0].dot(trying_dudrda[2][i][a][2]) + g_r[2].dot(trying_dudrda[2][i][a][0])) / two;
              // st
              Real deda_st_B = (g_r[1].dot(trying_dudrda[1][i][a][2]) + g_r[2].dot(trying_dudrda[1][i][a][1])) / two;
              Real deda_st_C = (g_r[1].dot(trying_dudrda[2][i][a][2]) + g_r[2].dot(trying_dudrda[2][i][a][1])) / two;

              // rt
              deda[i][a][0] = deda_rt_A + s * (deda_rt_C - deda_rt_A - deda_st_C + deda_st_B);
              // st
              deda[i][a][1] = deda_st_B - r * (deda_rt_C - deda_rt_A - deda_st_C + deda_st_B);
            }
          }

          for (int i = 0; i < 3; i++) { // v_id
            for (int j = 0; j < 3; j++) { // v_id
              for (int a = 0; a < 3; a++) { // dof
                for (int b = 0; b < 3; b++) { // dof
                  Real tmp = 0.;
                  for (int m = 0; m < 2; m++) { // natural coord (rt, st).
                    for (int n = 0; n < 2; n++) { // natural coord (rt, st).
                      tmp += c[m][n] * deda[i][a][m] * deda[j][b][n];
                    }
                  }
                  ret[0](3 * i + a, 3 * j + b) += tri_area * thick * weight * tmp;
                }
              }
            }
          }

          // derivatives
          for (int k = 0; k < 3; k++) {
            for (int xy = 0; xy < 2; xy++) {
              Real dedadx[3][3][2];
              for (int i = 0; i < 3; i++)  { // v_id
                for (int a = 0; a < 3; a++) { // dof
                  // rt
                  Real dedadx_rt_A = (dg_r[k][xy][0].dot(trying_dudrda[0][i][a][2]) + dg_r[k][xy][2].dot(trying_dudrda[0][i][a][0])) / two;
                  Real dedadx_rt_C = (dg_r[k][xy][0].dot(trying_dudrda[2][i][a][2]) + dg_r[k][xy][2].dot(trying_dudrda[2][i][a][0])) / two;
                  // st
                  Real dedadx_st_B = (dg_r[k][xy][1].dot(trying_dudrda[1][i][a][2]) + dg_r[k][xy][2].dot(trying_dudrda[1][i][a][1])) / two;
                  Real dedadx_st_C = (dg_r[k][xy][1].dot(trying_dudrda[2][i][a][2]) + dg_r[k][xy][2].dot(trying_dudrda[2][i][a][1])) / two;

                  // rt
                  dedadx[i][a][0] = dedadx_rt_A + s * (dedadx_rt_C - dedadx_rt_A - dedadx_st_C + dedadx_st_B);
                  // st
                  dedadx[i][a][1] = dedadx_st_B - r * (dedadx_rt_C - dedadx_rt_A - dedadx_st_C + dedadx_st_B);
                }
              }

              for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                  for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                      Real tmp = 0;
                      Real dtmp = 0;
                      for (int m = 0; m < 2; m++) {
                        for (int n = 0; n < 2; n++) {
                          tmp += c[m][n] * deda[i][a][m] * deda[j][b][n];
                          dtmp += (dc[k][xy][m][n] * deda[i][a][m] * deda[j][b][n]) + (c[m][n] * dedadx[i][a][m] * deda[j][b][n]) + (c[m][n] * deda[i][a][m] * dedadx[j][b][n]);
                        }
                      }
                      ret[2 * k + xy + 1](3 * i + a, 3 * j + b) += (dtri_area[2 * k + xy] * thick / two * tmp) + (tri_area * thick / two * dtmp);
                    }
                  }
                }
              }
            }
          }
        } // integration loop
      } // integration for rt, st

      return ret;
    }

    std::array<std::array<Real, 9>, 7> elem_mass_diags(Id_t n0, Id_t n1, Id_t n2)
    {
      std::array<std::array<Real, 9>, 7> ret;

      auto x0 = vtx_buffer[2 * n0 + 0];
      auto y0 = vtx_buffer[2 * n0 + 1];
      auto x1 = vtx_buffer[2 * n1 + 0];
      auto y1 = vtx_buffer[2 * n1 + 1];
      auto x2 = vtx_buffer[2 * n2 + 0];
      auto y2 = vtx_buffer[2 * n2 + 1];

      auto tri_area = calc_area(x0, y0, x1, y1, x2, y2);
      auto dtri_area = calc_darea(x0, y0, x1, y1, x2, y2);

      auto three = Real(3.);
      auto four = Real(4.);

      auto mab = tri_area / three * rho * thick * thick * thick / (three * four);
      auto mz  = tri_area / three * rho * thick;

      for (int i = 0; i < 3; i++) {
        ret[0][3 * i + 0] = mab;
        ret[0][3 * i + 1] = mab;
        ret[0][3 * i + 2] = mz;
      }

      // derivatives
      for (int k = 0; k < 3; k++) {
        for (int xy = 0; xy < 2; xy++) {
          auto dmab = dtri_area[2 * k + xy] / three * rho * thick * thick * thick / (three * four);
          auto dmz  = dtri_area[2 * k + xy] / three * rho * thick;

          for (int i = 0; i < 3; i++) {
              ret[2 * k + xy + 1][3 * i + 0] = dmab;
              ret[2 * k + xy + 1][3 * i + 1] = dmab;
              ret[2 * k + xy + 1][3 * i + 2] = dmz;
          }
        }
      }

      return ret;
    }

    void calc_stiff_matrix()
    {
      stiff_matrix->clear_values();

      for (auto& deriv : stiff_derivatives) {
        deriv->clear_values();
      }

      for (int t = 0; t < num_triangle; t++) {
        Id_t n0 = idx_buffer[3 * t + 0];
        Id_t n1 = idx_buffer[3 * t + 1];
        Id_t n2 = idx_buffer[3 * t + 2];
        auto elem_stiff = elem_stiff_matrix(n0, n1, n2);

        Id_t ids[3] = { n0, n1, n2 };
        for (int i = 0; i < 3; i++) { // v0
          for (int j = 0; j < 3; j++) { // v1
            stiff_matrix->add_local_block<9, 9>(ids[i], ids[j], 3 * i, 3 * j, elem_stiff[0]);
            // add derivatives
            for (int k = 0; k < 3; k++) {
              for (int xy = 0; xy < 2; xy++) {
                // non-zero derivative
                if (ids[k] < num_edge_vtx)
                  stiff_derivatives[2 * ids[k] + xy]->add_local_block<9, 9>(ids[i], ids[j], 3 * i, 3 * j, elem_stiff[2 * k + xy + 1]);
              }
            }
          }
        }
      }
    }

    void calc_mass_diags()
    {
      // clear values
      std::fill(mass_diags.begin(), mass_diags.end(), 0.);

      for (auto& deriv : mass_derivatives) {
        deriv->clear_values();
      }

      for (int t = 0; t < num_triangle; t++) {
        Id_t n0 = idx_buffer[3 * t + 0];
        Id_t n1 = idx_buffer[3 * t + 1];
        Id_t n2 = idx_buffer[3 * t + 2];
        auto elem_mass = elem_mass_diags(n0, n1, n2);

        Id_t ids[3] = { n0, n1, n2 };
        for (int i = 0; i < 3; i++) {
          mass_diags[3 * ids[i] + 0] += elem_mass[0][3 * i + 0];
          mass_diags[3 * ids[i] + 1] += elem_mass[0][3 * i + 1];
          mass_diags[3 * ids[i] + 2] += elem_mass[0][3 * i + 2];
        }

        // derivatives
        for (int k = 0; k < 3; k++) {
          for (int xy = 0; xy < 2; xy++) {
            for (int i = 0; i < 3; i++) {
              // non-zero derivative
              if (ids[k] < num_edge_vtx)
                mass_derivatives[2 * ids[k] + xy]->add_block(ids[i], {
                  elem_mass[2 * k + xy + 1][3 * i + 0],
                  elem_mass[2 * k + xy + 1][3 * i + 1],
                  elem_mass[2 * k + xy + 1][3 * i + 2]
                });
            }
          }
        }
      }
    }

    void calc_eig_val_derivatives(Real eig_val, const std::vector<Real>& eig_vec)
    {
      eig_val_derivatives.resize(num_vtx * 2, 0.);
      for (int i = 0; i < num_edge_vtx * 2; i++) {
        auto uMu = diag_quadratic(mass_diags, eig_vec);
        auto sdMdx = eig_val * (*mass_derivatives[i]);
        auto dKdx_sdMdx = bsr_bs_diag_sub(*stiff_derivatives[i], sdMdx);
        auto udKdx_sdMdxu = bsr_quadratic(dKdx_sdMdx, eig_vec);
        eig_val_derivatives[i] = udKdx_sdMdxu / uMu;
      }
    }

    void calc_whole_area()
    {
      // reset
      whole_area = 0.;

      for (int t = 0; t < num_triangle; t++) {
        Id_t n0 = idx_buffer[3 * t + 0];
        Id_t n1 = idx_buffer[3 * t + 1];
        Id_t n2 = idx_buffer[3 * t + 2];
        auto x0 = vtx_buffer[2 * n0 + 0];
        auto y0 = vtx_buffer[2 * n0 + 1];
        auto x1 = vtx_buffer[2 * n1 + 0];
        auto y1 = vtx_buffer[2 * n1 + 1];
        auto x2 = vtx_buffer[2 * n2 + 0];
        auto y2 = vtx_buffer[2 * n2 + 1];

        auto tri_area = calc_area(x0, y0, x1, y1, x2, y2);
        whole_area += tri_area;
      }
    }

    // copy
    void update_vtx_buffer(const std::vector<Real>& vtx_buffer_)
    { vtx_buffer = vtx_buffer_; }

    // getter
    auto get_stiff_matrix() const -> const BsrMatrix<3, 3>&
    { return *stiff_matrix; }

    auto get_mass_diags() const -> const std::vector<Real>&
    { return mass_diags; }

    auto get_whole_area() const -> Real
    { return whole_area; }

    auto get_eig_val_derivatives() const -> const std::vector<Real>&
    { return eig_val_derivatives; }

    auto get_graph_laplacian() const -> const CsrMatrix<Real>&
    { return graph_laplacian; }

  private:
    // constants
    Real thick, lambda, myu, rho;

    Real whole_area = 0;

    int num_vtx, num_edge_vtx, num_triangle;

    std::vector<Id_t> idx_buffer;
    std::vector<Real> vtx_buffer;

    std::unique_ptr<BsrMatrix<3, 3>> stiff_matrix;
    std::vector<Real> mass_diags;

    // derivatives
    std::vector<std::unique_ptr<BsrMatrix<3, 3>>>  stiff_derivatives;
    std::vector<std::unique_ptr<BsArray<Real, 3>>> mass_derivatives;
    std::vector<Real> eig_val_derivatives;

    CsrMatrix<Real> graph_laplacian;
};

} // namespace Mitc3