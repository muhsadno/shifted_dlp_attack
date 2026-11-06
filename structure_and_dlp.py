"""
Structure of X_{y,g} = {y g^k} and recovery of the secret exponent.

Krylov isomorphism: span{T^i y} ~= F_8[x]/(mu), with T^k y <-> x^k mod mu.
Writing mu = x^e * mu' (x not dividing mu'):
    index  n = e,     period  l = ord(x mod mu').
Recovering a from A = y g^a is then an ordinary DLP of x in (F_8[x]/mu')^*,
solved here by Pohlig-Hellman (+ BSGS per prime).
"""
import math
from functools import reduce
import numpy as np
import galois
import sympy
from platform_port import GF
from krylov_attack import solve_in_basis

def P(coeffs_low_to_high):
    return galois.Poly(GF(list(coeffs_low_to_high)[::-1]) if len(coeffs_low_to_high) else GF([0]), field=GF)

XP = galois.Poly([1, 0], field=GF)
ONEP = galois.Poly([1], field=GF)

def min_poly(blocks, nxt):
    m = solve_in_basis(blocks, nxt)                  # T^d y = sum m_i T^i y
    d = len(blocks)
    return P(list(-m) + [GF(1)])                     # x^d - sum m_i x^i

def mult_order(h, mod, group_order):
    """Order of h in (F_8[x]/mod)^*, given a multiple 'group_order' of it."""
    o = group_order
    for q in sympy.factorint(group_order):
        while o % q == 0 and pow(h, o // q, mod) == ONEP:
            o //= q
    return o

def structure(mu):
    facs, mults = mu.factors()
    e = 0
    parts = []                    # (f, k, ord of x mod f^k)
    for f, k in zip(facs, mults):
        if f == XP:
            e = k
            continue
        m = f.degree
        o = mult_order(XP % f, f, 8**m - 1)
        # 2-part from repeated factors: x^o = 1 + f*(...) has 2-power order mod f^k
        fk = f**k
        t = 0
        while pow(XP, o * 2**t, fk) != ONEP:
            t += 1
        parts.append((f, k, o * 2**t))
    mu_prime = reduce(lambda a, b: a * b, [f**k for f, k, _ in parts], ONEP)
    l = reduce(math.lcm, [o for *_, o in parts], 1)
    return e, l, parts, mu_prime

def key(p):
    return tuple(int(c) for c in p.coeffs)

def bsgs(h, t, q, mod):
    """Solve h^z = t mod 'mod' with 0 <= z < q."""
    msz = math.isqrt(q) + 1
    table, cur = {}, ONEP
    for j in range(msz):
        table.setdefault(key(cur), j)
        cur = (cur * h) % mod
    factor = pow(h, q - msz, mod) if q >= msz else None   # h^{-msz} since h^q = 1
    cur = t % mod
    for i in range(msz + 1):
        if key(cur) in table:
            return (i * msz + table[key(cur)]) % q
        cur = (cur * factor) % mod
    raise ValueError("BSGS failed")

def pohlig_hellman(r, l, mod):
    """Find z mod l with x^z = r in (F_8[x]/mod)^*, x of order l."""
    residues, moduli = [], []
    for q, v in sympy.factorint(l).items():
        qv = q**v
        h = pow(XP, l // q, mod)              # order q
        z = 0
        for j in range(v):
            # remove known digits: (r * x^{-z})^{l/q^{j+1}}
            tj = pow((r * pow(XP, (l - z) % l, mod)) % mod, l // q**(j + 1), mod)
            dj = bsgs(h, tj, q, mod)
            z += dj * q**j
        residues.append(z); moduli.append(qv)
    z, _ = sympy.ntheory.modular.crt(moduli, residues)
    return int(z)

def recover_exponent(blocks, nxt, A_rows):
    mu = min_poly(blocks, nxt)
    e, l, parts, mu_prime = structure(mu)
    c = solve_in_basis(blocks, A_rows)            # A = r_a(T) y
    r_a = P(list(c))
    z = pohlig_hellman(r_a % mu_prime, l, mu_prime)
    # lift to an exponent >= e (lands in the cycle like the real secret)
    a_rec = z + l * max(0, -(-(e - z) // l))
    assert pow(XP, a_rec, mu) == r_a % mu, "recovered exponent inconsistent"
    return dict(mu=mu, e=e, l=l, parts=parts, a_rec=a_rec)
