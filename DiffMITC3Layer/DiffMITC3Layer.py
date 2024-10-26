import numpy as np
import scipy
import torch


def create_graph_laplacian(plate, num_vtx) :
    laplacian_indptr  = plate.get_laplacian_indptr()
    laplacian_indices = plate.get_laplacian_indices()
    laplacian_data    = plate.get_laplacian_data()
    return scipy.sparse.csr_matrix((laplacian_data, laplacian_indices, laplacian_indptr), shape=[num_vtx, num_vtx])

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

    # TODO : backward
    @staticmethod
    def backward(ctx, dLdeval, dLdevec):
        dLdx = np.zeros((ctx.num_vtx, 2))
        for i in range(ctx.num_eig_ids) :
            ctx.plate.calc_backward(ctx.evals[i], ctx.evecs[i])
            devaldx = ctx.plate.get_eig_val_derivative()
            dLdx = np.add(dLdx, dLdeval[i] * devaldx.reshape([ctx.num_vtx, 2]))
        return dLdx, None, None

def calc_tone_loss(vtx_buffer, plate, target_freqs, freqs_weights) :
    eig_val, eig_vec = SolveMITC3LinearSystem.apply(vtx_buffer, plate, target_freqs.shape[0])
    freqs = torch.sqrt(eig_val) / (2. * np.pi)
    loss_tone = torch.dot(freqs_weights, torch.pow(freqs - target_freqs, 2) / target_freqs)
    return loss_tone, freqs