from scipy import sparse
import numpy as np
from tqdm import tqdm

import fastplotlib as fpl

from project import SpMVImage

adapter = fpl.enumerate_adapters()[0]
print(adapter.info.device)
fpl.select_adapter(adapter)

# test with some random data
m, n = (512, 512)
p = m * n

k = 1_000
T = 250

A_csr: sparse.csr_matrix = sparse.random(
    p, k, density=0.1, format="csr", dtype=np.float32
)
C = np.random.rand(k, T).astype(np.float32)


spmv_img = SpMVImage(
    A_csr,
    C,
    shape=(m, n),
    benchmark=True,
    spmv_mode="scalar",  # set to "vector" or "scalar" the specific CSR mat-vec kernel
)

err = np.zeros(T)
# check that results match pytorch
for i in tqdm(range(T)):
    truth = (A_csr.dot(C[:, i])).reshape(m, n)

    spmv_img.t = i
    diff = spmv_img.to_numpy() - truth
    # compute relative error using Frobenius norm
    err[i] = np.linalg.norm(diff, ord="fro") / np.linalg.norm(truth, ord="fro")

print(err.max())
