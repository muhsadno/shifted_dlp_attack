"""
Linear-algebra (Krylov) attack on the shifted Diffie-Hellman protocol over
M_3(F_8[A_5]).

Right multiplication by g is an F_8-linear map T on the row space F_8[A_5]^3
(dimension 180), and every public value is y*g^k = T^k(y) applied row-wise.

Key recovery (no discrete log needed):
    1. Build the Krylov vectors y, Ty, T^2 y, ... until they become dependent.
    2. Solve  A = sum_i c_i T^i y   (linear system over F_8).
    3. Output sum_i c_i T^i B.
Since T commutes with every polynomial in T,
    sum_i c_i T^i B = r_a(T) T^b y = T^b r_a(T) y = T^b A = A g^b = K.
"""
import time
import numpy as np
import galois
from platform_port import GF, N, a_mul, a_mono

def right_mult_matrix(g):
    """180x180 matrix Tm over F_8 with  (row vector v in F_8[A5]^3) @ Tm = v * g."""
    Tm = GF.Zeros((3 * N, 3 * N))
    for j in range(3):
        for h in range(N):
            e = a_mono(h)
            row = GF.Zeros(3 * N)
            for k in range(3):
                row[k * N:(k + 1) * N] = a_mul(e, g[j, k])
            Tm[j * N + h] = row
    return Tm

def flat(P):                     # (3,3,60) -> (3,180): the three rows of P
    return P.reshape(3, 3 * N)

def krylov(y_rows, Tm, max_dim=3 * N + 1):
    """Return the list of Krylov blocks [y, yT, yT^2, ...] spanning a basis,
    plus the dimension d (first index where T^d y depends on earlier ones)."""
    blocks = [y_rows]
    cur = y_rows
    for d in range(1, max_dim + 1):
        cur = cur @ Tm
        M = GF(np.stack([b.reshape(-1) for b in blocks + [cur]], axis=1))
        if np.linalg.matrix_rank(M) == len(blocks):     # cur is dependent
            return blocks, cur, d
        blocks.append(cur)
    raise RuntimeError("Krylov space did not close")

def solve_in_basis(blocks, target):
    """Find c with target = sum c_i blocks[i] (over F_8)."""
    Kmat = GF(np.stack([b.reshape(-1) for b in blocks], axis=1))      # (540, d)
    aug = GF(np.hstack([Kmat, target.reshape(-1, 1)]))
    R = aug.row_reduce()
    d = Kmat.shape[1]
    # pivot columns of the first d columns are 0..d-1 since Kmat has full column rank
    if np.any(R[d:, :]):
        raise ValueError("target not in Krylov space")
    return R[:d, d]

def attack(inst, verbose=True):
    # t0 = time.time()
    t0 = time.perf_counter()
    Tm = right_mult_matrix(inst["g"])
    Y, A, B = flat(inst["y"]), flat(inst["A"]), flat(inst["B"])
    blocks, _, d = krylov(Y, Tm)
    c = solve_in_basis(blocks, A)
    # apply r_a(T) to B
    K_rec = GF.Zeros(B.shape)
    cur = B
    for i in range(d):
        if c[i]:
            K_rec += c[i] * cur
        cur = cur @ Tm
    ok = np.array_equal(K_rec, flat(inst["K"]))
    # t = time.time() - t0
    t = time.perf_counter() - t0
    if verbose:
        print(f"  Krylov dimension d = {d}, attack time {t:.2f}s, shared key recovered: {ok}")
    return ok, d, t, Tm, blocks, c
