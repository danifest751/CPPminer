# gen_examples.py OUT.sass : produce cuobjdump-style examples of dual-carry IADD3.X forms via nvdisasm (oracle)
import random
import subprocess
import sys


def dis(lo, hi):
    open("/tmp/o.bin", "wb").write(lo.to_bytes(8, "little") + hi.to_bytes(8, "little"))
    r = subprocess.run(["nvdisasm", "-b", "SM75", "/tmp/o.bin"], capture_output=True, text=True)
    out = [l for l in r.stdout.split("\n") if "/*0000*/" in l]
    return out[0].split("/*0000*/")[1].split(";")[0].strip() if out else None


def setf(v, pos, width, val):
    mask = ((1 << width) - 1) << pos
    return (v & ~mask) | ((val << pos) & mask)


random.seed(1)
# base: IADD3.X R5, P0, R5, R7, RZ, P0, !PT   lo 0x0000000705057210 hi 0x000fe4000071e4ff
# lo: Rd bits 16-23, Ra 24-31, Rb 32-39 ; hi: Rc 0-7, Pin2 13-15, Pout1 17-19, Pout2 20-22, Pin1 23-25 (probe)
base_lo, base_hi = 0x0000000705057210, 0x000fe4000071e4ff
ex = []
seen = set()
for i in range(600):
    lo, hi = base_lo, base_hi
    lo = setf(lo, 16, 8, random.randrange(64))
    lo = setf(lo, 24, 8, random.randrange(64))
    lo = setf(lo, 32, 8, random.choice([random.randrange(64), 255]))
    hi = setf(hi, 0, 8, random.choice([random.randrange(64), 255]))
    hi = setf(hi, 17, 3, random.randrange(7))
    hi = setf(hi, 20, 3, random.choice([7, random.randrange(7)]))
    hi = setf(hi, 13, 3, random.randrange(7))
    hi = setf(hi, 23, 3, random.randrange(7))
    hi = setf(hi, 16, 1, random.choice([0, 0, 0, 1]))   # Pin2 negate
    hi = setf(hi, 26, 1, random.choice([0, 0, 0, 1]))   # Pin1 negate (probe)
    t = dis(lo, hi)
    if i < 8:
        print("raw:", t, hex(lo), hex(hi))
    if t and t.startswith("IADD3.X") and t not in seen:
        seen.add(t)
        ex.append((t, lo, hi))
print(len(ex), "examples; sample:", ex[:4])
with open(sys.argv[1], "w") as f:
    f.write("\n\tcode for sm_75\n\t\tFunction : fake_kernel\n\t.headerflags\t@\"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)\"\n")
    for i, (t, lo, hi) in enumerate(ex):
        f.write("        /*%04x*/                   %s ;  /* 0x%016x */\n" % (i * 16, t, lo))
        f.write("                                                                                 /* 0x%016x */\n" % hi)
    f.write("\t\t..........\n")
