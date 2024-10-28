import numpy as np
import scipy
import torch
from del_msh import PolyLoop
import DiffMITC3Impl


def create_graph_laplacian(plate, num_vtx) :
    laplacian_indptr  = plate.get_laplacian_indptr()
    laplacian_indices = plate.get_laplacian_indices()
    laplacian_data    = plate.get_laplacian_data()
    return scipy.sparse.csr_matrix((laplacian_data, laplacian_indices, laplacian_indptr), shape=[num_vtx, num_vtx])


class TessellateEdgeVtx(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, edge_vtx_buffer, resolution) :
        edge_vtx_buffer = edge_vtx_buffer.reshape((-1, 2)).detach().numpy().astype(np.float32)
        ctx.num_edge_vtx = edge_vtx_buffer.shape[0]
        idx_buffer, vtx_buffer, _ = PolyLoop.tesselation2d(edge_vtx_buffer, resolution_edge=-1, resolution_face=resolution)
        idx_buffer = torch.from_numpy(idx_buffer.astype(np.int32))
        vtx_buffer = torch.from_numpy(vtx_buffer.astype(np.float64))
        vtx_buffer.requires_grad = True
        return idx_buffer, vtx_buffer

    @staticmethod
    def backward(ctx, dLidx, dLdvtx):
        dLdedge = torch.zeros([ctx.num_edge_vtx, 2], dtype=torch.float64, requires_grad=True)
        dLdedge = dLdedge.copy_(dLdvtx[0 : ctx.num_edge_vtx])
        return dLdedge, None

class SolveMITC3LinearSystem(torch.autograd.Function) :
    @staticmethod
    # vtx_buffer = [x0, y0, x1, y1, ... ]
    # only one eigen value for optimization
    def forward(ctx, vtx_buffer, plate, num_eig_ids) :
        ctx.plate = plate
        ctx.num_vtx = vtx_buffer.shape[0]
        ctx.num_eig_ids = num_eig_ids

        vtx_buffer = vtx_buffer.reshape([-1])
        plate.update_vtx_buffer(vtx_buffer)
        plate.calc_forward()

        # convert to sparse matrix
        stiff_indptr  = plate.get_stiff_indptr()
        stiff_indices = plate.get_stiff_indices()
        stiff_data    = plate.get_stiff_data().reshape([-1, 3, 3])
        mass_diags    = plate.get_mass_diags()

        smat = scipy.sparse.bsr_matrix((stiff_data, stiff_indices, stiff_indptr), shape=[ctx.num_vtx * 3, ctx.num_vtx * 3])
        mmat = scipy.sparse.dia_matrix((mass_diags.reshape(1, -1), np.array([0])), shape=[ctx.num_vtx * 3, ctx.num_vtx * 3])

        evals, evecs = scipy.sparse.linalg.eigsh(smat, k=num_eig_ids + 3, M=mmat, sigma=0, maxiter=500)
        # ignore dof of rotations
        evals = evals[3:].real
        evecs = evecs.transpose()[3:].real

        ctx.evals = evals
        ctx.evecs = evecs
        # eigen values and eigen vectors
        return torch.from_numpy(evals), torch.from_numpy(evecs)

    @staticmethod
    def backward(ctx, dLdeval, dLdevec):
        dLdx = np.zeros((ctx.num_vtx, 2))
        for i in range(ctx.num_eig_ids) :
            ctx.plate.calc_backward(ctx.evals[i], ctx.evecs[i])
            devaldx = ctx.plate.get_eig_val_derivative()
            dLdx = np.add(dLdx, dLdeval[i] * devaldx.reshape([ctx.num_vtx, 2]))
        return dLdx, None, None


# TODO : select device
class Plate :
    def __init__(self, resolution, thickness_, lambda_, myu_, rho_, edge_vtx_buffer):
        self.num_edge_vtx = edge_vtx_buffer.shape[0]
        self.idx_buffer, self.vtx_buffer = TessellateEdgeVtx.apply(edge_vtx_buffer, resolution)
        self.num_vtx = self.vtx_buffer.shape[0]
        self.previous_vtx = self.vtx_buffer.detach().clone()
        self.plate = DiffMITC3Impl.MITC3Plate(thickness_, lambda_, myu_, rho_, self.num_vtx, self.num_edge_vtx, self.idx_buffer.reshape(-1))
        self.laplacian = create_graph_laplacian(self.plate, self.num_vtx)
        self.bilaplacian = self.laplacian * self.laplacian
        self.smoothing_diag = scipy.sparse.dia_matrix((np.ones(self.num_vtx), np.array([0])), shape=[self.num_vtx, self.num_vtx])
        self.scale = torch.tensor(1., requires_grad=True)

    def get_scaled_vtx(self) :
        return self.scale * self.vtx_buffer

    def calc_tone_loss(self, target_freqs, freqs_weights) :
        vtx_buffer = self.get_scaled_vtx().cpu()
        eig_val, eig_vec = SolveMITC3LinearSystem.apply(vtx_buffer, self.plate, target_freqs.shape[0])
        freqs = torch.sqrt(eig_val) / (2. * np.pi)
        loss_tone = torch.dot(freqs_weights, torch.pow(freqs - target_freqs, 2) / target_freqs)
        return loss_tone, freqs

    def laplacian_smoothing(self, laplacian_weight) :
        with torch.no_grad() :
            smoothed_dvtx = scipy.sparse.linalg.spsolve(
                laplacian_weight * self.bilaplacian + self.smoothing_diag,
                (self.vtx_buffer - self.previous_vtx).detach().cpu().numpy()
            )
            self.vtx_buffer.copy_(self.previous_vtx + torch.from_numpy(smoothed_dvtx))
            self.vtx_buffer.requires_grad = True
            self.previous_vtx = self.vtx_buffer.detach().clone()
