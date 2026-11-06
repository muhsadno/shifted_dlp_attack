import random, time, json, sys, numpy as np, sympy
from platform_port import *
from krylov_attack import attack, flat
from structure_and_dlp import recover_exponent
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def warmup():
    random.seed(9999999)
    inst = key_exchange()
    ok, d, t_att, Tm, blocks, c = attack(inst, verbose=False)
    assert ok, "CDH warm-up failed"
    # Warm-up DLP attack
    nxt = blocks[-1] @ Tm
    res = recover_exponent(
        blocks, nxt, flat(inst["A"])
    )
    assert (res["a_rec"] - inst["a"]) % res["l"] == 0, (
        "DLP warm-up failed"
    )
    print("Warm-up complete.", flush=True)

warmup()

ntrials = int(sys.argv[1])
rows = []
for trial in range(ntrials):
    # random.seed(1000 + trial)
    random.seed(2026 + trial)
    # t = time.time(); inst = key_exchange(); t_ke = time.time() - t
    t = time.perf_counter(); inst = key_exchange(); t_ke = time.perf_counter() - t
    ok, d, t_att, Tm, blocks, c = attack(inst, verbose=False)

    # eavesdropper decrypts an AES-CBC message keyed as in the notebook
    iv = get_random_bytes(16)
    ct = AES.new(derive_aes_key(inst["K"]), AES.MODE_CBC, iv).encrypt(pad(b"Super Secret Message!!!", 16))
    K_rec = GF.Zeros((3,3,N)); 
    # rebuild the recovered key matrix
    cur = flat(inst["B"]); Krows = GF.Zeros(cur.shape)
    for i in range(d):
        if c[i]: Krows += c[i]*cur
        cur = cur @ Tm
    pt = unpad(AES.new(derive_aes_key(Krows.reshape(3,3,N)), AES.MODE_CBC, iv).decrypt(ct), 16)

    # secret-exponent recovery (what a forger of the signature would need)
    # t = time.time()
    t = time.perf_counter()
    nxt = blocks[-1] @ Tm
    res = recover_exponent(blocks, nxt, flat(inst["A"]))
    # t_dlp = time.time() - t
    t_dlp = time.perf_counter() - t
    l, e = res["l"], res["e"]
    a_ok = (res["a_rec"] - inst["a"]) % l == 0
    degs = sorted([(f.degree, k) for f, k, _ in res["parts"]], reverse=True)
    lp = max(sympy.factorint(l)) if l > 1 else 1
    row = dict(trial=trial, krylov_dim=d, key_recovered=bool(ok), plaintext=pt.decode(),
               index_e=e, period_bits=round(np.log2(float(l)),1), period=str(l),
               factor_degrees=degs, largest_prime_of_period_bits=round(np.log2(float(lp)),1),
               a_recovered_mod_l=bool(a_ok), t_keyexchange=round(t_ke,1), t_key_attack=round(t_att,2),
               t_dlp=round(t_dlp,2))
    if trial == 0:
        # direct check in the semigroup itself: y * g^{a_rec} == A
        row["direct_check_y_g^a_rec==A"] = bool(np.array_equal(m_mul(inst["y"], m_pow(inst["g"], res["a_rec"])), inst["A"]))
    print(json.dumps(row), flush=True)
    rows.append(row)
json.dump(rows, open("results.json","w"), indent=1)