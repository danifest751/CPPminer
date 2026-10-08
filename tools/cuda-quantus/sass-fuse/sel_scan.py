"""sel_scan.py in.cuasm KERNEL: classify the consumers of carry materializations
`SEL Rm, RZ, 0x1, !Pc` (Rm = Pc ? 1 : 0) in one kernel."""
import re
import sys
from collections import Counter

from sassdf import Ins

src, ksym = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
is_sec = lambda l: re.match(r"\s*\.section\s", l) is not None
start = next(i for i, l in enumerate(lines) if is_sec(l) and ".text." + ksym in l)
end = next(i for i in range(start + 1, len(lines)) if is_sec(lines[i]))
ins = [x for x in (Ins(lines[i], i) for i in range(start, end)) if x.ok]
shapes = Counter()
for k, x in enumerate(ins):
    if not (x.op == "SEL" and not x.gpred and len(x.ops) == 4 and x.ops[1] == "RZ" and x.ops[2] == "0x1"
            and re.fullmatch(r"!P\d", x.ops[3])):
        continue
    rm, pc = x.wregs[0], x.ops[3][1:]
    uses = []
    pc_redef = None
    for j in range(k + 1, len(ins)):
        y = ins[j]
        if y.op.startswith("BRA") or y.op == "EXIT":
            uses.append(("<branch>", j))
            break
        if rm in y.rregs:
            uses.append((y, j))
        if pc_redef is None and pc in y.wpreds:
            pc_redef = j
        if rm in y.wregs:
            break
    for y, j in uses:
        if y == "<branch>":
            shapes["live across branch"] += 1
            continue
        shape = re.sub(r"R\d+", "R", y.op + " " + ", ".join(y.ops))
        shape = shape.replace(rm.replace("R", "R"), "R")
        pc_ok = pc_redef is None or pc_redef >= j
        shapes["%s | %d uses | Pc %s" % (shape, len(uses), "kept" if pc_ok else "clobbered")] += 1
for s, n in shapes.most_common(40):
    print("%4d  %s" % (n, s))
