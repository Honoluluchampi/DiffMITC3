#include "pybind_util.hpp"
#include "Mitc3.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
class Mitc3PlateWrapper
{
  public:
    explicit Mitc3PlateWrapper(
      const Real thick_,
      const Real lambda_,
      const Real myu_,
      const Real rho_,
      const int num_vtx_,
      const int num_edge_vtx_,
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& idx_buffer)
      : plate(thick_, lambda_, myu_, rho_, num_vtx_, num_edge_vtx_, pyarray2vector(idx_buffer)) {}

    void update_vtx_buffer(const py::array_t<Real, py::array::c_style | py::array::forcecast>& vtx_buffer)
    { plate.update_vtx_buffer(pyarray2vector(vtx_buffer)); }

    void calc_forward()
    {
      plate.calc_stiff_matrix();
      plate.calc_mass_diags();
    }

    void calc_backward(
      const Real eig_val,
      const py::array_t<Real, py::array::c_style | py::array::forcecast>& eig_vec)
    { plate.calc_eig_val_derivatives(eig_val, pyarray2vector(eig_vec)); }

    auto get_stiff_indptr() const -> py::array_t<uint32_t>
    { return vector2pyarray(plate.get_stiff_matrix().get_indptr()); }

    auto get_stiff_indices() const -> py::array_t<uint32_t>
    { return vector2pyarray(plate.get_stiff_matrix().get_indices()); }

    auto get_stiff_data() const -> py::array_t<Real>
    { return vector2pyarray(plate.get_stiff_matrix().get_data()); }

    auto get_mass_diags() const -> py::array_t<Real>
    { return vector2pyarray(plate.get_mass_diags()); }

    auto get_eig_val_derivatives() const -> py::array_t<Real>
    { return vector2pyarray(plate.get_eig_val_derivatives()); }

    auto get_laplacian_indptr() const -> py::array_t<uint32_t>
    { return vector2pyarray(plate.get_graph_laplacian().indptr); }

    auto get_laplacian_indices() const -> py::array_t<uint32_t>
    { return vector2pyarray(plate.get_graph_laplacian().indices); }

    auto get_laplacian_data() const -> py::array_t<Real>
    { return vector2pyarray(plate.get_graph_laplacian().data); }

  private:
    Mitc3::Plate plate;
};

// -----------------------------------------------------------------------

PYBIND11_MODULE(Mitc3FEM, m) {
  pybind11::class_<Mitc3PlateWrapper>(m, "Mitc3Plate")
    .def(py::init<
      const Real, // thick
      const Real, // lambda
      const Real, // myu
      const Real, // rho
      const int,  // num_vtx
      const int,  // num_edge_vtx
      const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& // idx_buffer
    >())
    // forward
    .def("update_vtx_buffer", &Mitc3PlateWrapper::update_vtx_buffer, py::arg("vtx_buffer"))
    .def("calc_forward",      &Mitc3PlateWrapper::calc_forward, "calculates stiffness matrix and mass matrix")
    .def("get_mass_diags",    &Mitc3PlateWrapper::get_mass_diags,    py::return_value_policy::reference)
    .def("get_stiff_indptr",  &Mitc3PlateWrapper::get_stiff_indptr,  py::return_value_policy::reference, "indptr of BSR format")
    .def("get_stiff_indices", &Mitc3PlateWrapper::get_stiff_indices, py::return_value_policy::reference, "column indices of BSR format")
    .def("get_stiff_data",    &Mitc3PlateWrapper::get_stiff_data,    py::return_value_policy::reference, "data of BSR format")
    // backward
    .def("calc_backward", &Mitc3PlateWrapper::calc_backward, py::arg("eig_val"), py::arg("eig_vec"))
    .def("get_eig_val_derivative", &Mitc3PlateWrapper::get_eig_val_derivatives, py::return_value_policy::reference)
    // laplacian
    .def("get_laplacian_indptr",   &Mitc3PlateWrapper::get_laplacian_indptr,  py::return_value_policy::reference, "indptr of BSR format")
    .def("get_laplacian_indices",  &Mitc3PlateWrapper::get_laplacian_indices, py::return_value_policy::reference, "indices of BSR format")
    .def("get_laplacian_data",     &Mitc3PlateWrapper::get_laplacian_data,    py::return_value_policy::reference, "data of BSR format")
    ;
}