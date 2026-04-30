from scipy import sparse
import fastplotlib as fpl

from project import SpMVImage

import masknmf

# if WGPU can't find an adapter it will raise, or it will give you a LLVM CPU adapter which you don't want
# if you have multiple GPUs you can select it using the int index, here the GPU at index 0 is selected.
adapter = fpl.enumerate_adapters()[0]
print(adapter.info)
fpl.select_adapter(adapter)

# path to the demixing results data file
dmr_path = "./demix.hdf5"
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)

# create the scipy CSR sparse arrays
# denoise results
V = dmr.pmd_array.v.cpu().numpy()
U = dmr.pmd_array.u
U_csr_torch = U.to_sparse_csr().cpu()
U_csr = sparse.csr_matrix(
    (U_csr_torch.values(), U_csr_torch.col_indices(), U_csr_torch.crow_indices()),
    shape=U_csr_torch.size(),
)

# demix results
C = dmr.c.numpy().T
A = dmr.a
A_torch_csr = dmr.a.to_sparse_csr().cpu()
A_csr = sparse.csr_matrix(
    (A_torch_csr.values(), A_torch_csr.col_indices(), A_torch_csr.crow_indices()),
    shape=A_torch_csr.size(),
)

m, n = tuple(map(int, dmr.fov_shape))
p = m * n
k = int(V.shape[0])
T = int(V.shape[1])

# scale factor and addition that must be added to re-scale the frames for denoising reconstruction
scale_factor = dmr.pmd_array.var_img.ravel().cpu().numpy()
scale_add = dmr.pmd_array.mean_img.ravel().cpu().numpy()

# create objects that render with the data from teh compute kernels
pmd = SpMVImage(
    U_csr,
    V,
    scale_factor=scale_factor,
    scale_add=scale_add,
    shape=(m, n),
    spmv_mode="vector",  # choose the kernel
)

ac = SpMVImage(
    A_csr,
    C,
    shape=(m, n),
    spmv_mode="scalar",
)

fig = fpl.Figure(
    shape=(1, 2),
    names=["pmd", "ac"],
    controller_ids="sync",
    size=(1200, 700),
    canvas_kwargs={"max_fps": 999, "vsync": False},
)

fig["pmd"].add_graphic(pmd.image_graphic)
fig["ac"].add_graphic(ac.image_graphic)

fig["pmd"].tooltip.enabled = False

def update(figure):
    t = pmd.t
    t += 1

    if t == T:
        t = 0

    pmd.t = t
    ac.t = t


fig.add_animations(update)
fig.imgui_show_fps = True
fig.show()
fpl.loop.run()
