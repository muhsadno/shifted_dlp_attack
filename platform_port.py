"""
Faithful Python port of the platform in brayendamaraa/shifted-diffie-hellman
(shifted-dh.ipynb): 3x3 matrices over the group algebra F_8[A_5], with the
same sampling procedure for y and g and the same key exchange.

Conventions copied from SageMath:
  * GF(2^3, 'x') uses the Conway polynomial x^3 + x + 1 (galois' default too).
  * Permutation products are left-to-right: (p*q)(i) = q(p(i)).
"""
import itertools
import random
import numpy as np
import galois

GF = galois.GF(2**3)          # x^3 + x + 1
X = GF(2)                     # the generator 'x' of F_8

# ---------- A_5 -------------------------------------------------------------
def _sign(p):
    s, seen = 1, [False] * 5
    for i in range(5):
        if not seen[i]:
            j, L = i, 0
            while not seen[j]:
                seen[j] = True; j = p[j]; L += 1
            s *= (-1) ** (L - 1)
    return s

ELEMS = [p for p in itertools.permutations(range(5)) if _sign(p) == 1]
IDX = {p: i for i, p in enumerate(ELEMS)}
N = len(ELEMS)                # 60
ID = IDX[tuple(range(5))]

def compose(p, q):            # Sage convention: apply p, then q
    return tuple(q[p[i]] for i in range(5))

MT = np.array([[IDX[compose(p, q)] for q in ELEMS] for p in ELEMS])

def perm_from_cycles(*cycles):
    """Sage-style 1-based cycle notation, e.g. perm_from_cycles((1,3),(2,5))."""
    img = list(range(5))
    for c in cycles:
        for k in range(len(c)):
            img[c[k] - 1] = c[(k + 1) % len(c)] - 1
    return tuple(img)

# ---------- group algebra F_8[A_5] -------------------------------------------
def a_zero():
    return GF.Zeros(N)

def a_mono(i, coeff=1):
    v = GF.Zeros(N); v[i] = GF(coeff) if not isinstance(coeff, GF) else coeff
    return v

def a_mul(a, b):
    c = GF.Zeros(N)
    for i in np.nonzero(a)[0]:
        c[MT[i]] += a[i] * b       # MT[i] is a permutation of 0..59
    return c

def a_random():
    # return GF.Random(N)
    return GF([random.randrange(8) for _ in range(N)])

# ---------- 3x3 matrices over the algebra: shape (3, 3, 60) ------------------
def m_mul(P, Q):
    R = GF.Zeros((3, 3, N))
    for i in range(3):
        for k in range(3):
            if not np.any(P[i, k]):
                continue
            for j in range(3):
                R[i, j] += a_mul(P[i, k], Q[k, j])
    return R

def m_identity():
    I = GF.Zeros((3, 3, N))
    for i in range(3):
        I[i, i, ID] = 1
    return I

def m_pow(P, e):
    R, B = m_identity(), P.copy()
    while e:
        if e & 1:
            R = m_mul(R, B)
        e >>= 1
        if e:
            B = m_mul(B, B)
    return R

def m_scalar_left(s, P):       # s * P  (Sage: scalar acting on the left)
    R = GF.Zeros((3, 3, N))
    for i in range(3):
        for j in range(3):
            R[i, j] = a_mul(s, P[i, j])
    return R

# ---------- the notebook's sampling ------------------------------------------
x = X
ONE = GF(1)
_terms = [
    (x**2,             perm_from_cycles((3, 4, 5))),
    (x**2 + x,         perm_from_cycles((2, 5, 4))),
    (x**2 + x + ONE,     perm_from_cycles((1, 3, 5, 2, 4))),
    (x,                perm_from_cycles((1, 3), (2, 5))),
    (x + ONE,            perm_from_cycles((1, 4, 3, 5, 2))),
    (x**2 + x + ONE,     perm_from_cycles((1, 4), (2, 3))),
    (x,                perm_from_cycles((1, 5, 4, 2, 3))),
    (GF(1),            perm_from_cycles((1, 5, 2, 3, 4))),
]
S_ELT = a_zero()
for c, p in _terms:
    S_ELT[IDX[p]] += c

def random_invertible_triangular_matrix(pos):
    M = GF.Zeros((3, 3, N))
    for i in range(3):
        rng = range(i, 3) if pos == "upper" else range(i + 1)
        for j in rng:
            M[i, j] = a_mono(random.randrange(N)) if i == j else a_random()
    return M

def random_invertible_matrix():
    M = m_identity()
    for _ in range(20):
        M = m_mul(M, random_invertible_triangular_matrix(random.choice(["upper", "lower"])))
    return M

def random_non_invertible_matrix():
    return m_scalar_left(S_ELT, random_invertible_matrix())

def is_idempotent(P):
    return np.array_equal(m_mul(P, P), P)

# ---------- key derivation (same recipe as the notebook) ---------------------
def derive_aes_key(Kmat):
    bits = "".join(format(int(c), "03b") for c in Kmat[0, 0])   # 60 coeffs * 3 bits
    return int(bits[:128], 2).to_bytes(16, "big")

def key_exchange(secret_bits=100):
    y = random_non_invertible_matrix()
    while is_idempotent(y):
        y = random_non_invertible_matrix()
    g = random_non_invertible_matrix()
    a = random.randint(1, 2**secret_bits)
    b = random.randint(1, 2**secret_bits)
    A = m_mul(y, m_pow(g, a))
    B = m_mul(y, m_pow(g, b))
    K = m_mul(A, m_pow(g, b))
    return dict(y=y, g=g, a=a, b=b, A=A, B=B, K=K)
