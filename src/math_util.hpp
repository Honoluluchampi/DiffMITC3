#pragma once

#include <cassert>
#include <set>
#include <unordered_map>
#include <iostream>

#include "../submodules/eigen/Eigen/Dense"

using Real = double;

template <int Row, int Col>
using Matrix = Eigen::Matrix<Real, Row, Col>;

// hash function for std::pair
template<class T> size_t HashCombine(const size_t seed,const T &v){
  return seed^(std::hash<T>()(v)+0x9e3779b9+(seed<<6)+(seed>>2));
}

template<class T,class S> struct std::hash<std::pair<T,S>>{
  size_t operator()(const std::pair<T,S> &keyval) const noexcept {
    return HashCombine(std::hash<T>()(keyval.first), keyval.second);
  }
};

// compatible with scipy's bsr matrix format
template <int Row, int Col>
class BsrMatrix
{
  public:
    template <typename Indptr, typename Indices, typename ColOffset>
    explicit BsrMatrix(
      Indptr&& indptr_,
      Indices&& indices_,
      ColOffset&& col_offsets_,
      int row_size_, int col_size_)
    {
      // forward adjacency information
      indptr = std::forward<Indptr>(indptr_);
      indices = std::forward<Indices>(indices_);
      col_offsets = std::forward<ColOffset>(col_offsets_);

      block_size = Row * Col;
      row_size = row_size_;
      col_size = col_size_;

      data_size = indptr[indptr.size() - 1] * block_size;
      data.resize(data_size, 0);
    }

    explicit BsrMatrix(const std::vector<std::vector<uint32_t>>& adj_arrays, int row_size_, int col_size_)
    {
      row_size = row_size_;
      col_size = col_size_;
      block_size = Row * Col;

      // TODO : reserve unordered_map
      col_offsets.reserve(row_size * 5);
      indptr.resize(1, 0);
      indices.resize(0);

      for (uint32_t i = 0; i < row_size; i++) {
        int acc_sum = 0;
        for (uint32_t j = 0; j < adj_arrays[i].size(); j++) {
          acc_sum += 1;
          col_offsets.emplace(std::pair{i, adj_arrays[i][j]}, acc_sum);
          indices.push_back(adj_arrays[i][j]);
        }
        indptr.push_back(indptr[i] + acc_sum);
      }

      data_size = indptr[indptr.size() - 1] * block_size;
      data.resize(data_size, 0.);
    }

    BsrMatrix(const BsrMatrix&) = default;
    BsrMatrix& operator=(const BsrMatrix&) = default;
    BsrMatrix(BsrMatrix&&) = default;
    BsrMatrix& operator=(BsrMatrix&&) = default;

    // r and c is the indices of block
    void add_block(int r, int c, const Matrix<Row, Col>& block)
    {
      int offset = (indptr[r] + col_offsets.at({r, c}) - 1) * block_size;
      for (int i = 0; i < Row; i++) {
        for (int j = 0; j < Col; j++) {
          data[offset + i * Col + j] += block(i, j);
        }
      }
    }

    template <int GlobalR, int GlobalC>
    void add_local_block(int r, int c, int local_r, int local_c, const Matrix<GlobalR, GlobalC>& block)
    {
      if (col_offsets.find({r, c}) == col_offsets.end()) {
        std::cerr << "invalid id for bsr matrix" << std::endl;
        return;
      }
      int offset = (indptr[r] + col_offsets.at({r, c}) - 1) * block_size;
      for (int i = 0; i < Row; i++) {
        for (int j = 0; j < Col; j++) {
          data[offset + i * Col + j] += block(local_r + i, local_c + j);
        }
      }
    }

    void diag_add(int r, int c, int size, const Real* block)
    {
      int offset = (indptr[r] + col_offsets.at({r, c}) - 1) * block_size;
      for (int i = 0; i < size; i++) {
        data[offset + i * Col + i] += block[i];
      }
    }

    void diag_sub(int r, int c, int size, const Real* block)
    {
      int offset = (indptr[r] + col_offsets.at({r, c}) - 1) * block_size;
      for (int i = 0; i < size; i++) {
        data[offset + i * Col + i] -= block[i];
      }
    }

    void clear_values()
    { std::fill(data.begin(), data.end(), 0.); }

    auto get_indptr() const -> const std::vector<uint32_t>&
    { return indptr; }

    auto get_indices() const -> const std::vector<uint32_t>&
    { return indices; }

    auto get_data() const -> const std::vector<Real>&
    { return data; }

    auto get_shape() const -> std::pair<uint32_t, uint32_t>
    { return { row_size * Row, col_size * Col }; }


    void print() const
    {
      std::cout << "indptr" << std::endl;
      for (const auto& r : indptr) std::cout << r << " ";
      std::cout << std::endl;
      std::cout << "indices" << std::endl;
      for (const auto& c : indices) std::cout << c << " ";
      std::cout << std::endl;
      std::cout << "data";
      for (int i = 0; i < data.size(); i++) {
        if (i % block_size == 0) std::cout << std::endl << "block " << i/block_size << std::endl;
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    }

  private:
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> col_offsets;
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<Real> data;
    // whole matrix size
    int row_size, col_size;
    int block_size;
    int data_size;
};

void add_adjacency(
  std::vector<std::vector<uint32_t>>& adj_arrays,
  std::unordered_map<std::pair<uint32_t, uint32_t>, bool>& adj_map,
  uint32_t n0, uint32_t n1, uint32_t n2, uint32_t n3)
{
  auto lambda = [&adj_map, &adj_arrays](uint32_t a, uint32_t b) {
    if (adj_map.find({a, b}) == adj_map.end()) {
      adj_map.emplace(std::pair{a, b}, true);
      adj_arrays[a].push_back(b);
    }
  };

  lambda(n0, n0); lambda(n0, n1); lambda(n0, n2); lambda(n0, n3);
  lambda(n1, n0); lambda(n1, n1); lambda(n1, n2); lambda(n1, n3);
  lambda(n2, n0); lambda(n2, n1); lambda(n2, n2); lambda(n2, n3);
  lambda(n3, n0); lambda(n3, n1); lambda(n3, n2); lambda(n3, n3);
}

template <int Row, int Col>
auto make_symmetric_bsr(
  uint32_t axis_id,
  const std::vector<uint32_t>& whole_indptr,
  const std::vector<uint32_t>& whole_indices,
  const std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>& whole_col_offset)
  -> BsrMatrix<Row, Col>
{
  std::vector<uint32_t> indptr(whole_indptr.size(), 0);
  std::vector<uint32_t> indices;
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> col_offset;
  int num_vtx = whole_indptr.size() - 1;

  // mean connected edge count
  indices.reserve(whole_indices.size() / num_vtx);

  for (int r = 0; r < num_vtx; r++) {
    int num_valid_elem = 0;

    if (whole_col_offset.find({axis_id, r}) != whole_col_offset.end()) {
      for (int ci = 0; ci < whole_indptr[r + 1] - whole_indptr[r]; ci++) {
        auto c = whole_indices[whole_indptr[r] + ci];
        if (whole_col_offset.find({axis_id, c}) != whole_col_offset.end()) {
          indices.emplace_back(c);
          col_offset.emplace(std::pair(r, c), num_valid_elem + 1);
          num_valid_elem++;
        }
      }
    }
    indptr[r + 1] = indptr[r] + num_valid_elem;
  }

  return BsrMatrix<Row, Col>(
    std::move(indptr),
    std::move(indices),
    std::move(col_offset),
    num_vtx, num_vtx
  );
}

// TODO : delete
auto symmetrize_adjacency(
  uint32_t id,
  const std::vector<std::vector<uint32_t>>& adj_arrays,
  const std::unordered_map<std::pair<uint32_t, uint32_t>, bool>& adj_map)
  -> std::vector<std::vector<uint32_t>>
{
  std::vector<std::vector<uint32_t>> ret(adj_arrays.size(), std::vector<uint32_t>());
  for (int i = 0; i < adj_arrays.size(); i++) {
    for (int j = 0; j < adj_arrays[i].size(); j++) {
      if (adj_map.find({id, i}) != adj_map.end() && adj_map.find({id, adj_arrays[i][j]}) != adj_map.end()) {
        ret[i].push_back(adj_arrays[i][j]);
      }
    }
  }

  for (int i = 0; i < ret.size(); i++) {
    std::sort(ret[i].begin(), ret[i].end());
  }

  return ret;
}

// block sparse array -----------------------------------------------------------
template <typename T, int Block_Size>
class BsArray
{
  public:
    explicit BsArray(const std::vector<uint32_t>& adj_array, uint32_t max_size_)
    {
      max_size = max_size_;
      col_offset.reserve(5);
      for (int i = 0; i < adj_array.size(); i++) {
        col_offset.emplace(adj_array[i], i + 1);
        indices.push_back(adj_array[i]);
      }
      data.resize(adj_array.size() * Block_Size, 0);
    }

    // from BsrMatrix adjacency information
    explicit BsArray(const std::vector<uint32_t>& whole_indices, uint32_t indptr0, uint32_t indptr1, uint32_t max_size_)
    {
      max_size = max_size_;
      auto num_elem = indptr1 - indptr0;

      // copy indices
      indices.resize(num_elem);
      std::copy(whole_indices.begin() + indptr0, whole_indices.begin() + indptr1, indices.begin());
      // set col offsets
      col_offset.reserve(num_elem);
      for (int i = 0; i < num_elem; i++) {
        col_offset.emplace(whole_indices[indptr0 + i], i + 1);
      }

      data.resize(num_elem * Block_Size, 0);
    }

    void add_block(uint32_t block_id, const std::array<T, Block_Size>& block)
    {
      if (col_offset.find(block_id) == col_offset.end()) {
        std::cerr << "invalid id for sparse array." << std::endl;
        return; // reject the invalid id
      }

      for (int i = 0; i < Block_Size; i++) {
        data[(col_offset.at(block_id) - 1) * Block_Size + i] += block[i];
      }
    }

    void add_uniform_block(uint32_t block_id, T value)
    {
      if (col_offset.find(block_id) == col_offset.end()) {
        std::cerr << "invalid id for sparse array." << std::endl;
        return; // reject the invalid id
      }

      for (int i = 0; i < Block_Size; i++) {
        data[(col_offset.at(block_id) - 1) * Block_Size + i] += value;
      }
    }

    template <int N>
    void add_multiple_uniform_block(const std::array<uint32_t, N>& ids, T value)
    {
      for (auto block_id : ids) {
        if (col_offset.find(block_id) == col_offset.end()) {
          std::cerr << "invalid block id for block array" << std::endl;
          return; // reject the invalid id
        }

        for (int i = 0; i < Block_Size; i++) {
          data[(col_offset.at(block_id) - 1) * Block_Size + i] += value;
        }
      }
    }

    void clear_values()
    { data.resize(data.size(), 0); }

    auto get_indices() const -> const std::vector<uint32_t>&
    { return indices; }

    auto get_data() const -> const std::vector<T>&
    { return data; }

    auto get_ith_block_ptr(int i) const -> const T*
    { return data.data() + Block_Size * i; }

    auto get_mut_data() -> std::vector<T>&
    { return data;}

    auto get_max_size() const -> uint32_t
    { return max_size; }

  private:
    std::unordered_map<uint32_t, uint32_t> col_offset;
    std::vector<uint32_t> indices;
    std::vector<T> data;
    uint32_t max_size;
};

template <typename T, int BlockSize>
BsArray<T, BlockSize> operator*(T lhs, const BsArray<T, BlockSize>& rhs)
{
  auto ret = rhs;
  for (T& data : ret.get_mut_data()) {
    data *= lhs;
  }
  return ret;
}

// compressed sparse array ----------------------------------------------------

auto det33(const Matrix<3, 3>& m) -> Real
{ return m(0,0)*m(1,1)*m(2,2) + m(0,1)*m(1,2)*m(2,0) + m(0,2)*m(1,0)*m(2,1) - m(0,2)*m(1,1)*m(2,0) - m(0,1)*m(1,0)*m(2,2) - m(0,0)*m(1,2)*m(2,1); }

auto inv33(const Matrix<3, 3>& m) -> Matrix<3, 3>
{
  Matrix<3, 3> ret;
  ret << m(1,1)*m(2,2) - m(1,2)*m(2,1), m(0,2)*m(2,1) - m(0,1)*m(2,2), m(0,1)*m(1,2) - m(0,2)*m(1,1),
         m(1,2)*m(2,0) - m(1,0)*m(2,2), m(0,0)*m(2,2) - m(0,2)*m(2,0), m(0,2)*m(1,0) - m(0,0)*m(1,2),
         m(1,0)*m(2,1) - m(1,1)*m(2,0), m(0,1)*m(2,0) - m(0,0)*m(2,1), m(0,0)*m(1,1) - m(0,1)*m(1,0);
  return 1. / det33(m) * ret;
}

auto bsr_bs_diag_sub(const BsrMatrix<3, 3>& lhs, const BsArray<Real, 3>& rhs) -> BsrMatrix<3, 3>
{
  BsrMatrix<3, 3> ret = lhs;
  const auto& br_indices = rhs.get_indices();

  for (int i = 0; i < br_indices.size(); i++) {
    auto idx = br_indices[i];
    ret.diag_sub(idx, idx, 3, rhs.get_ith_block_ptr(i));
  }

  return ret;
}

auto diag_quadratic(const std::vector<Real>& diag, const std::vector<Real>& vec) -> Real
{
  assert(diag.size() == vec.size() && "diag-mat size and vec size must be the same.");

  Real ret = 0;
  for (int i = 0; i < diag.size(); i++) {
    ret += diag[i] * vec[i] * vec[i];
  }

  return ret;
}

template <int BlockSize>
auto bs_diag_quadratic(const BsArray<Real, BlockSize>& diag_mat, const std::vector<Real>& vec) -> Real
{
  assert(diag_mat.get_max_size() == vec.size() && "mat-size and vec-size must be the same.");
  const auto& indices = diag_mat.get_indices();
  const auto& data = diag_mat.get_data();

  Real ret = 0;
  for (int i = 0; i < indices.size(); i++) {
    for (int j = 0; j < BlockSize; j++) {
      ret += data[BlockSize * i + j] * std::pow(vec[BlockSize * indices[i] + j], 2);
    }
  }

  return ret;
}

template <int RowColSize>
auto bsr_quadratic(const BsrMatrix<RowColSize, RowColSize>& mat, const std::vector<Real>& vec) -> Real
{
  auto block_size = RowColSize * RowColSize;
  const auto& indptr = mat.get_indptr();
  const auto& indices = mat.get_indices();
  const auto& data = mat.get_data();

  Real ret = 0;
  for (int i = 0; i < indptr.size() - 1; i++) {
    Real aij_uj[RowColSize] = { 0., 0., 0. };
    for (int j = indptr[i]; j < indptr[i + 1]; j++) {
      for (int k = 0; k < RowColSize; k++) {
        for (int l = 0; l < RowColSize; l++) {
          aij_uj[k] += data[block_size * j + RowColSize * k + l] * vec[RowColSize * indices[j] + l];
        }
      }
    }
    for (int j = 0; j < RowColSize; j++) {
      ret += aij_uj[j] * vec[RowColSize * i + j];
    }
  }

  return ret;
}

template <typename T>
struct CsrMatrix
{
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<T> data;
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> col_offsets;
};

// returns [indptr, indices]
auto make_graph_laplacian(const std::vector<uint32_t>& idx_buffer, uint32_t num_vtx) -> CsrMatrix<Real>
{
  assert(idx_buffer.size() % 3 == 0 && "size of idx buffer is invalid.");

  // scratch pad for construction
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> tmp_col_offsets;
  std::vector<std::vector<std::pair<uint32_t, Real>>> tmp_ind_data(num_vtx);

  for (int t = 0; t < idx_buffer.size() / 3; t++) {
    uint32_t ids[3] = {
      idx_buffer[t * 3 + 0],
      idx_buffer[t * 3 + 1],
      idx_buffer[t * 3 + 2]
    };

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (i == j)
          continue;

        std::pair<uint32_t, uint32_t> key = { ids[i], ids[j] };
        // not registered yet
        if (tmp_col_offsets.find(key) == tmp_col_offsets.end()) {
          tmp_col_offsets.emplace(key, tmp_ind_data[ids[i]].size());
          tmp_ind_data[ids[i]].emplace_back(ids[j], -1);
          // check diag components
          std::pair<uint32_t, uint32_t> diag_key = { ids[i], ids[i] };
          if (tmp_col_offsets.find(diag_key) == tmp_col_offsets.end()) {
            tmp_col_offsets.emplace(diag_key, tmp_ind_data[ids[i]].size());
            tmp_ind_data[ids[i]].emplace_back(ids[i], 1);
          }
          else {
            tmp_ind_data[ids[i]][tmp_col_offsets.at(diag_key)].second += 1;
          }
        }
      }
    }
  }

  // sort
  int data_size = 0;
  for (int i = 0; i < num_vtx; i++) {
    std::sort(tmp_ind_data[i].begin(), tmp_ind_data[i].end());
    data_size += tmp_ind_data[i].size();
  }

  std::vector<uint32_t> indptr(num_vtx + 1, 0);
  std::vector<uint32_t> indices(data_size);
  std::vector<Real> data(data_size);
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> col_offsets;
  col_offsets.reserve(data_size);

  for (int i = 0; i < num_vtx; i++) {
    indptr[i + 1] = indptr[i] + tmp_ind_data[i].size();
    for (int j = 0; j < tmp_ind_data[i].size(); j++) {
      auto id = tmp_ind_data[i][j].first;
      indices[indptr[i] + j] = id;
      data[indptr[i] + j] = tmp_ind_data[i][j].second;
      col_offsets.emplace(std::pair<uint32_t, uint32_t>(i, id), j + 1);
    }
  }

  CsrMatrix<Real> ret;
  ret.indptr  = std::move(indptr);
  ret.indices = std::move(indices);
  ret.data    = std::move(data);
  ret.col_offsets = std::move(col_offsets);

  return ret;
}