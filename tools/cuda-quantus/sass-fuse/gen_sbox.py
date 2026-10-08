"""gen_sbox.py in.cuasm out.cuasm [--sqr dup|mov] [--yield N] [--order rr|blk]
Replace the xor placeholders in k_hand's loop with hand-written SASS for x^7 mod p on every chain.

Arithmetic per chain (32-bit words, EPS = 2^32 - 1 from c[0x3][0x0]):
  gmul: P=a0*b0, Q=a1*b1, O=a0*b1, O+=a1*b0 (carry pB); w1=P1+O0 (pA); w2=Q0+O1+pA (pA);
        w3=Q1+pA+pB  -> (Q1:Q0:P1:P0)
  gsqr: P=a0^2, Q=a1^2, O=a0*a1; P1 += 2*O0 (dual carry), Q0 += 2*O1 + carries (dual), Q1 += carries
  red : T = Q0*EPS + (P1:P0) (carry pA); z0 = T0 + ~Q1 + !pA (carry pB); z1 = T1 - 1 + pA + pB
        = (T1:T0) + pA*2^32 - (Q1 + pA), the same value as the production red128.
Stall counts come from a list schedule with the latencies ptxas itself uses on sm_75."""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
opt = lambda name, d: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else d
SQR = opt("--sqr", "mov")
YIELD = int(opt("--yield", "4"))
ORDER = opt("--order", "rr")
RED = opt("--red", "mine")      # mine | ptx
SQX = opt("--sqx", "dual")      # dual | single
KSYM = "k_hand"
EPS = "c[0x3][0x0]"

lines = open(src).read().split("\n")
is_sec = lambda l: re.match(r"\s*\.section\s", l) is not None
start = next(i for i, l in enumerate(lines) if is_sec(l) and ".text." + KSYM in l)
end = next(i for i in range(start + 1, len(lines)) if is_sec(lines[i]))
ph_re = re.compile(r"^(\s*)\[(B[-0-9]+):R(.):W(.):(.):S(\d\d)\](\s*)/\*([0-9a-f]+)\*/(\s*)LOP3\.LUT (R\d+), (R\d+), 0x1111(\d\d)(\d\d), RZ, 0x3c, !PT ;")
ph = {}
for i in range(start, end):
    m = ph_re.match(lines[i])
    if m:
        assert m[10] == m[11]
        ph[(int(m[12], 16), int(m[13], 16))] = (i, m)
nch = len(ph) // 2
assert nch >= 1 and len(ph) == 2 * nch, ph.keys()
reg_line = next(i for i in range(start, end) if "SHI_REGISTERS=" in lines[i])
base = int(re.search(r"SHI_REGISTERS=(\d+)", lines[reg_line])[1])
base += base & 1
print("chains:", nch, "placeholder regs:", {k: v[1][10] for k, v in sorted(ph.items())}, "scratch from R%d" % base)


class I:
    def __init__(self, op, args, w, r, cls):
        self.op, self.args, self.w, self.r, self.cls = op, args, w, r, cls

    def text(self):
        return "%s %s" % (self.op, self.args)


def R(n):
    return "R%d" % n


def chain_program(c):
    xl = int(ph[(c, 0)][1][10][1:])
    xh = int(ph[(c, 1)][1][10][1:])
    S = base + 16 * c
    x = (xl, xh)
    x2, x3, x4 = (S, S + 1), (S + 2, S + 3), (S + 4, S + 5)
    Pp, Qp, Op, Tp, Mp = (S + 6, S + 7), (S + 8, S + 9), (S + 10, S + 11), (S + 12, S + 13), (S + 14, S + 15)
    pA, pB = "P%d" % (1 + 2 * c), "P%d" % (2 + 2 * c)
    prog = []

    def wide(d, a, b, acc=None, pout=None, xin=None):
        args = [R(d[0])] + ([pout] if pout else []) + [R(a), b if b.startswith("c[") else b, R(acc[0]) if acc else "RZ"]
        if xin:
            args.append(xin)
        rr = [R(a)] + ([b] if not b.startswith("c[") else []) + ([R(acc[0]), R(acc[1])] if acc else []) + ([xin] if xin else [])
        prog.append(I("IMAD.WIDE.U32" + (".X" if xin else ""), ", ".join(args), [R(d[0]), R(d[1])] + ([pout] if pout else []), rr, "WIDE"))

    def alu(op, args, w, r):
        prog.append(I(op, args, w, r, "ALU"))

    def mov(d, s):
        prog.append(I("IMAD.MOV.U32", "%s, RZ, RZ, %s" % (R(d), R(s)), [R(d)], [R(s)], "IMAD"))

    def red(z):
        wide(Tp, Qp[0], EPS, acc=Pp, pout=pA)
        if RED == "ptx":
            # ptxas form: rc = Q1 + pA (FMA pipe); z0 = T0 - rc -> pB ; z1 = T1 - 1 + pB + pA
            prog.append(I("IMAD.X", "%s, RZ, RZ, %s, %s" % (R(Qp[1]), R(Qp[1]), pA), [R(Qp[1])], [R(Qp[1]), pA], "IMAD"))
            alu("IADD3", "%s, %s, -%s, %s, RZ" % (R(z[0]), pB, R(Qp[1]), R(Tp[0])), [R(z[0]), pB], [R(Qp[1]), R(Tp[0])])
            alu("IADD3.X", "%s, RZ, -0x1, %s, %s, %s" % (R(z[1]), R(Tp[1]), pB, pA), [R(z[1])], [R(Tp[1]), pA, pB])
            return
        # T = Q0*EPS + (P1:P0) -> pA ; z0 = T0 + ~Q1 + !pA -> pB ; z1 = T1 - 1 + pA + pB
        alu("IADD3.X", "%s, %s, %s, ~%s, RZ, !%s, !PT" % (R(z[0]), pB, R(Tp[0]), R(Qp[1]), pA), [R(z[0]), pB], [R(Tp[0]), R(Qp[1]), pA])
        alu("IADD3.X", "%s, %s, -0x1, RZ, %s, %s" % (R(z[1]), R(Tp[1]), pA, pB), [R(z[1])], [R(Tp[1]), pA, pB])

    def gmul(a, b, z):
        wide(Pp, a[0], R(b[0]))
        wide(Qp, a[1], R(b[1]))
        wide(Op, a[0], R(b[1]))
        wide(Op, a[1], R(b[0]), acc=Op, pout=pB)
        alu("IADD3", "%s, %s, %s, %s, RZ" % (R(Pp[1]), pA, R(Pp[1]), R(Op[0])), [R(Pp[1]), pA], [R(Pp[1]), R(Op[0])])
        alu("IADD3.X", "%s, %s, %s, %s, RZ, %s, !PT" % (R(Qp[0]), pA, R(Qp[0]), R(Op[1]), pA), [R(Qp[0]), pA], [R(Qp[0]), R(Op[1]), pA])
        alu("IADD3.X", "%s, %s, RZ, RZ, %s, %s" % (R(Qp[1]), R(Qp[1]), pA, pB), [R(Qp[1])], [R(Qp[1]), pA, pB])
        red(z)

    def gsqr(a, z):
        wide(Pp, a[0], R(a[0]))
        wide(Qp, a[1], R(a[1]))
        wide(Op, a[0], R(a[1]))
        if SQR == "mov":
            mov(Mp[0], Op[0])
            mov(Mp[1], Op[1])
            l2, h2 = Mp
        else:
            l2, h2 = Op
        alu("IADD3", "%s, %s, %s, %s, %s, %s" % (R(Pp[1]), pA, pB, R(Pp[1]), R(Op[0]), R(l2)), [R(Pp[1]), pA, pB], [R(Pp[1]), R(Op[0]), R(l2)])
        if SQX == "dual":
            alu("IADD3.X", "%s, %s, %s, %s, %s, %s, %s, %s" % (R(Qp[0]), pA, pB, R(Qp[0]), R(Op[1]), R(h2), pA, pB),
                [R(Qp[0]), pA, pB], [R(Qp[0]), R(Op[1]), R(h2), pA, pB])
        else:
            # one carry-out per .X: Q0 + O1 + pA -> pA, then + h2 + pB -> pB
            alu("IADD3.X", "%s, %s, %s, %s, RZ, %s, !PT" % (R(Qp[0]), pA, R(Qp[0]), R(Op[1]), pA), [R(Qp[0]), pA], [R(Qp[0]), R(Op[1]), pA])
            alu("IADD3.X", "%s, %s, %s, %s, RZ, %s, !PT" % (R(Qp[0]), pB, R(Qp[0]), R(h2), pB), [R(Qp[0]), pB], [R(Qp[0]), R(h2), pB])
        alu("IADD3.X", "%s, %s, RZ, RZ, %s, %s" % (R(Qp[1]), R(Qp[1]), pA, pB), [R(Qp[1])], [R(Qp[1]), pA, pB])
        red(z)

    body = opt("--body", "sbox")
    if body == "sqr":        # x <- x^2
        gsqr(x, x)
    elif body == "mul":      # x <- x^2 through the general product
        gmul(x, x, x)
    elif body == "mul3":     # x <- x^3
        gsqr(x, x2)
        gmul(x2, x, x)
    else:
        gsqr(x, x2)
        gmul(x2, x, x3)
        gsqr(x2, x4)
        gmul(x3, x4, x)
    return prog, S + 16


progs = []
maxreg = 0
for c in range(nch):
    p, top = chain_program(c)
    if "--limit" in sys.argv:
        p = p[:int(opt("--limit", "0"))]
    if "--probe-reg" in sys.argv:   # identity body plus one write to a high register
        hr = int(opt("--probe-reg", "64"))
        p = [I("IADD3", "R%d, %s, 0x1, RZ" % (hr, ph[(c, 0)][1][10]), ["R%d" % hr], [ph[(c, 0)][1][10]], "ALU")]
        top = hr + 2
    elif "--identity" in sys.argv:
        p = [I("LOP3.LUT", "%s, %s, 0x%08x, RZ, 0x3c, !PT" % (ph[(c, h)][1][10], ph[(c, h)][1][10], 0x11110000 + c * 256 + h),
               [ph[(c, h)][1][10]], [ph[(c, h)][1][10]], "ALU") for h in (0, 1)]
    progs.append(p)
    maxreg = max(maxreg, top)
# Declared registers must exceed the highest one used with some slack: on the CMP, R62 faulted
# with 64 declared and R64 with 66, R64 ran with 72. Round up to a multiple of 8 above top + 2.
maxreg = (maxreg + 2 + 7) // 8 * 8
if "--regs" in sys.argv:
    maxreg = int(opt("--regs", "0"))

# register-read rule (measured on sm_75): at most two reads per bank (index parity) per IADD3
for p in progs:
    for x in p:
        if x.op.startswith("IADD3"):
            regs = [r for r in x.r if r.startswith("R")]
            par = [int(r[1:]) % 2 for r in regs]
            assert par.count(0) <= 2 and par.count(1) <= 2, x.text()

# interleave
seq = []
if ORDER == "rr":
    for k in range(len(progs[0])):
        for p in progs:
            seq.append(p[k])
else:
    for p in progs:
        seq += p

LAT_W = {"WIDE": 6, "ALU": 4, "IMAD": 4}


def lat(p, c):
    if p == "WIDE":
        return 6 if c == "WIDE" else 5
    if p == "ALU":
        return 4 if c == "ALU" else 5
    return 5 if c == "ALU" else 4   # IMAD (mov)


issue = []
ready = {}      # reg -> (issue, cls) of last writer
for k, x in enumerate(seq):
    t = issue[-1] + 1 if issue else 0
    for r in x.r:
        if r in ready:
            pi, pc = ready[r]
            t = max(t, pi + lat(pc, x.cls))
    for r in x.w:
        if r in ready:
            pi, pc = ready[r]
            t = max(t, pi + LAT_W[pc] - LAT_W[x.cls] + 1)
    issue.append(t)
    for r in x.w:
        ready[r] = (t, x.cls)
stalls = [issue[k + 1] - issue[k] for k in range(len(seq) - 1)] + [8]
assert all(1 <= s <= 15 for s in stalls), max(stalls)
cycles = issue[-1] + 8
print("instructions: %d per iteration (%d per sbox), schedule length %d cycles/warp, %.1f issue slots per sbox"
      % (len(seq), len(seq) // nch, cycles, cycles / nch))

# splice
first = min(v[0] for v in ph.values())
waits = set()
for i, m in ph.values():
    for ch in m[2][1:]:
        if ch != "-":
            waits.add(ch)
bmask = "B" + "".join(str(n) if str(n) in waits else "-" for n in range(6))
m0 = ph[min(ph, key=lambda k: ph[k][0])][1]
out_block = []
for k, (x, s) in enumerate(zip(seq, stalls)):
    b = bmask if k == 0 else "B------"
    y = "Y" if s >= YIELD else "-"
    out_block.append("%s[%s:R-:W-:%s:S%02d]%s/*%s*/%s%s ;" % (m0[1], b, y, s, m0[7], m0[8], m0[9], x.text()))
drop = set(v[0] for v in ph.values())
# Instructions ptxas interleaved with the placeholders (the loop's ISETP) move into the block a few
# slots in, so the predicate they set has the whole block to settle before the loop's BRA reads it
# (ptxas spaces ISETP -> BRA by 12+ cycles; right before the BRA it was read stale).
last = max(drop)
foreign = [i for i in range(first + 1, last) if i not in drop and re.match(r"^\s*\[B", lines[i])]
pos = min(6, len(out_block))
out_block[pos:pos] = [lines[i] for i in foreign]
tail_cycles = sum(int(re.search(r":S(\d\d)\]", l)[1]) for l in out_block[pos + len(foreign):])
if tail_cycles < 13:
    out_block[-1] = re.sub(r":S\d\d\]", ":S%02d]" % (13 - tail_cycles + int(re.search(r":S(\d\d)\]", out_block[-1])[1])), out_block[-1])
drop |= set(foreign)
new = []
for i, l in enumerate(lines):
    if i == first:
        new += out_block
    elif i in drop:
        continue
    elif i == reg_line:
        new.append(re.sub(r"SHI_REGISTERS=\d+", "SHI_REGISTERS=%d" % maxreg, l))
    elif start <= i < reg_line and "__section_info" in l:
        # sh_info of a CUDA text section: register count in the top byte, symbol index below
        v = int(re.search(r"0x([0-9a-f]+)", l)[1], 16)
        new.append(re.sub(r"0x[0-9a-f]+", "0x%08x" % ((maxreg << 24) | (v & 0xFFFFFF)), l, count=1))
    else:
        new.append(l)
open(dst, "w").write("\n".join(new))
print("registers:", maxreg)
