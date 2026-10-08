"""fuse.py in.cuasm out.cuasm KERNEL_SYMBOL [--dry]
Post-ptxas peephole for sm_75: fuse pairs of dependent IADD3 / IADD3.X whose intermediate register is
dead into one 3-input IADD3 with up to two carry-outs and two carry-ins, then re-derive the stall
counts from a latency table. Only the named kernel's text section is touched."""
import re
import sys
from collections import defaultdict

from sassdf import Ins, reg_of, pred_of, nxt, opclass, latency_table

FMA_PIPE = ("IMAD", "IMAD.WIDE")
OBS = {}      # (producer class, consumer class, kind) -> min distance observed in the ptxas schedule
LAT_FLOOR = 0


def lat(pop, cop, kind):
    """issue cycles required between producer and consumer: the ptxas schedule's own minimum for
    that pair, else 6 for WIDE->WIDE, 5 across pipes, 4 within a pipe (sm_75 measurements)"""
    p, c = opclass(pop), opclass(cop)
    if (p, c, kind) in OBS and OBS[(p, c, kind)] < 99:
        v = OBS[(p, c, kind)]
    elif p == "IMAD.WIDE" and c == "IMAD.WIDE":
        v = 6
    elif (p in FMA_PIPE) != (c in FMA_PIPE):
        v = 5
    else:
        v = 4
    return max(v, LAT_FLOOR)


class Operand:
    """IADD3 source operand: text like R5, -R5, ~R5, RZ, 0x1, -0x1, c[0x3][0x0]"""
    def __init__(self, t):
        self.t = t.replace(".reuse", "")
        self.reg = reg_of(self.t)
        self.is_rz = self.reg == "RZ"
        self.is_reg = self.reg is not None and not self.is_rz
        self.is_imm = self.reg is None


def parse_iadd3(x):
    """-> (outs[list of P], srcs[3 Operand], ins[list of 'P'/'!P'] or None if not a plain IADD3/IADD3.X)"""
    if x.op not in ("IADD3", "IADD3.X") or x.gpred:
        return None
    ops = list(x.ops)
    dst = ops.pop(0)
    if reg_of(dst) is None or reg_of(dst) == "RZ":
        return None
    outs = []
    while ops and pred_of(ops[0]) and not ops[0].startswith("!") and ops[0] != "PT":
        outs.append(ops.pop(0))
    ins = []
    while ops and pred_of(ops[-1]):
        ins.insert(0, ops.pop())
    if len(ops) != 3:
        return None
    srcs = [Operand(t) for t in ops]
    if x.op == "IADD3.X":
        if len(ins) != 2:
            return None
        ins = [p for p in ins if p != "!PT"]
        if any(p.startswith("!") for p in ins):
            return None
    elif ins:
        return None
    return dst, outs, srcs, ins


def fmt(dst, outs, srcs, ins):
    op = "IADD3.X" if ins else "IADD3"
    a = [dst] + outs + [s.t for s in srcs]
    if ins:
        a += ins + ["!PT"] * (2 - len(ins))
    return op, ", ".join(a)


class Block:
    def __init__(self):
        self.ins = []
        self.label = None
        self.succ = []


def main():
    src, dst, ksym = sys.argv[1], sys.argv[2], sys.argv[3]
    dry = "--dry" in sys.argv
    global LAT_FLOOR
    if "--lat" in sys.argv:
        LAT_FLOOR = int(sys.argv[sys.argv.index("--lat") + 1])
    lines = open(src).read().split("\n")
    # locate the kernel text section
    is_sec = lambda l: re.match(r"\s*\.section\s", l) is not None
    start = next(i for i, l in enumerate(lines) if is_sec(l) and ".text." + ksym in l)
    end = next(i for i in range(start + 1, len(lines)) if is_sec(lines[i]))
    items = []   # (kind, lineidx, obj)
    ins_list = []
    for i in range(start, end):
        l = lines[i]
        m = re.match(r"^\s*(\.L_[\w]+):\s*$", l)
        if m:
            items.append(("label", i, m[1]))
            continue
        x = Ins(l, i)
        if x.ok:
            items.append(("ins", i, x))
            ins_list.append(x)
    OBS.update(latency_table(ins_list, by_consumer=True))
    # basic blocks
    blocks = []
    cur = Block()
    labels = {}
    for kind, i, obj in items:
        if kind == "label":
            if cur.ins:
                blocks.append(cur)
                cur = Block()
            cur.label = obj
            labels[obj] = cur
        else:
            obj.block = cur
            cur.ins.append(obj)
            if obj.op.startswith("BRA") or obj.op == "EXIT":
                blocks.append(cur)
                cur = Block()
    if cur.ins:
        blocks.append(cur)
    for bi, b in enumerate(blocks):
        last = b.ins[-1]
        if last.op.startswith("BRA"):
            m = re.search(r"\(\.?(L_\w+)\)", last.line)
            tgt = "." + m[1] if m else None
            if tgt in labels:
                b.succ.append(labels[tgt])
            if last.gpred and bi + 1 < len(blocks):
                b.succ.append(blocks[bi + 1])
        elif last.op == "EXIT":
            pass
        elif bi + 1 < len(blocks):
            b.succ.append(blocks[bi + 1])
    # liveness (regs + preds as one namespace)
    for b in blocks:
        b.use, b.defs = set(), set()
        for x in b.ins:
            for r in x.rregs + x.rpreds:
                if r not in b.defs:
                    b.use.add(r)
            for r in x.wregs + x.wpreds:
                b.defs.add(r)
        b.live_in, b.live_out = set(), set()
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            lo = set()
            for s in b.succ:
                lo |= s.live_in
            li = b.use | (lo - b.defs)
            if lo != b.live_out or li != b.live_in:
                b.live_out, b.live_in = lo, li
                changed = True
    # per-instruction live-after sets
    for b in blocks:
        live = set(b.live_out)
        for x in reversed(b.ins):
            x.live_after = set(live)
            live -= set(x.wregs + x.wpreds)
            live |= set(x.rregs + x.rpreds)

    # ---- fusion
    fused = 0
    touched = set()   # line indices whose .reuse flags must go
    why = defaultdict(int)
    allow_neg = "--neg" in sys.argv
    for b in blocks:
        for k2, i2 in enumerate(b.ins):
            if i2.deleted or i2.rb != "-" or i2.wb != "-":
                continue
            p2 = parse_iadd3(i2)
            if not p2:
                continue
            dst2, outs2, srcs2, ins2 = p2
            for si, s in enumerate(srcs2):
                if not s.is_reg or s.t != s.reg:   # the linking operand itself must be plain
                    continue
                rx = s.reg
                # producer: most recent write of rx in this block before i2
                k1 = None
                for k in range(k2 - 1, -1, -1):
                    if rx in b.ins[k].wregs:
                        k1 = k
                        break
                if k1 is None:
                    continue
                i1 = b.ins[k1]
                if i1.deleted or i1.rb != "-" or i1.wb != "-":
                    continue
                p1 = parse_iadd3(i1)
                if not p1:
                    why["producer not IADD3"] += 1
                    continue
                dst1, outs1, srcs1, ins1 = p1
                if dst1 != rx or i1.wregs != [rx]:
                    continue
                if rx in i2.live_after:
                    why["intermediate live after i2"] += 1
                    continue
                between = b.ins[k1 + 1:k2]
                if any(rx in y.rregs for y in between) or sum(1 for t in srcs2 if t.reg == rx) != 1:
                    why["intermediate read between"] += 1
                    continue
                # capacity
                regs1 = [t for t in srcs1 if not t.is_rz]
                regs2 = [t for j, t in enumerate(srcs2) if not t.is_rz and j != si]
                allsrc = regs1 + regs2
                outs = outs1 + outs2
                ins = ins1 + ins2
                if len(allsrc) > 3 or len(outs) > 2 or len(ins) > 2:
                    why["capacity"] += 1
                    continue
                negs = [t for t in allsrc if t.is_reg and t.t != t.reg]
                if negs and (not allow_neg or ins or any(t.t.startswith("~") for t in negs)):
                    why["negated operand"] += 1
                    continue
                imms = [t for t in allsrc if t.is_imm]
                if len(imms) > 1:
                    why["two immediates"] += 1
                    continue
                regs = [t for t in allsrc if not t.is_imm]
                if imms:
                    if len(regs) == 0:
                        continue
                    lay = [regs[0], imms[0]] + (regs[1:2] if len(regs) > 1 else [Operand("RZ")])
                else:
                    lay = regs + [Operand("RZ")] * (3 - len(regs))
                # Turing register file: at most two operand reads per bank (bank = index parity) per
                # instruction; a third same-bank read returns a wrong value (measured on the CMP).
                banks = [int(t.reg[1:]) % 2 for t in lay if t.is_reg]
                if banks.count(0) > 2 or banks.count(1) > 2:
                    why["bank conflict"] += 1
                    continue
                # ---- placement A: fused instruction at i2's slot (i1 moves down)
                i1_reads = set(i1.rregs + i1.rpreds)
                okA = not any(set(y.wregs + y.wpreds) & i1_reads for y in between)
                okA = okA and not any(any(p in y.rpreds or p in y.wpreds for y in between) or p in i2.wpreds or p in i2.rpreds for p in outs1)
                # ---- placement B: fused instruction at i1's slot (i2 moves up)
                i2_reads = set(i2.rregs + i2.rpreds) - {rx}
                i2_writes = set(i2.wregs + i2.wpreds)
                okB = not any(set(y.wregs + y.wpreds) & i2_reads for y in between)
                okB = okB and not any((set(y.rregs + y.rpreds) | set(y.wregs + y.wpreds)) & i2_writes for y in between)
                # Hoisting i2's reads above a scoreboard wait (e.g. for an LDC result) would read the
                # register before the load lands: no wait mask on i2 or anything between.
                okB = okB and all(y.bar == "B------" for y in between + [i2])
                if not okA and not okB:
                    why["operands clobbered between"] += 1
                    continue
                if "--only-nox" in sys.argv and ins:
                    continue
                if "--only-x" in sys.argv and not ins:
                    continue
                if "--max" in sys.argv and fused >= int(sys.argv[sys.argv.index("--max") + 1]):
                    continue
                if "--only" in sys.argv:
                    seen_n = getattr(main, "seen", 0) + 1
                    main.seen = seen_n
                    if seen_n != int(sys.argv[sys.argv.index("--only") + 1]):
                        continue
                op, args = fmt(dst2, outs, lay, ins)
                keep, drop = (i2, i1) if okA else (i1, i2)
                print("fuse #%d @%s%s: [%s %s] + [%s %s] -> %s %s" % (fused + 1, keep.addr, "A" if okA else "B", i1.op, ", ".join(i1.ops), i2.op, ", ".join(i2.ops), op, args))
                if not dry:
                    drop.deleted = True
                    keep.new_op, keep.new_args = op, args
                    keep.ops = [t.strip() for t in args.split(",")]
                    keep.op = op
                    keep.compute_rw()
                    touched.add(i1.idx)
                    touched.add(i2.idx)
                    if k1 > 0:
                        touched.add(b.ins[k1 - 1].idx)
                    if k2 > 0:
                        touched.add(b.ins[k2 - 1].idx)
                fused += 1
                break   # one fusion per i2
    print("fused pairs:", fused, "rejections:", dict(why))

    # ---- optional pipe rebalancing: carry-free adds onto the FMA pipe (IADD3.X -> IMAD.X, IADD3 -> IMAD.IADD)
    conv = 0
    if "--to-imad" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--to-imad") + 1])
        for b in blocks:
            for k, x in enumerate(b.ins):
                if x.deleted or hasattr(x, "new_op") or x.rb != "-" or x.wb != "-":
                    continue
                if x.op == "SEL" and not x.gpred and len(x.ops) == 4 and x.ops[1] == "RZ" and x.ops[2] == "0x1" and re.fullmatch(r"!P\d", x.ops[3]):
                    # carry materialization: SEL Rd, RZ, 0x1, !P  (Rd = P ? 1 : 0)  ->  IMAD.X Rd, RZ, RZ, RZ, P
                    if conv >= limit:
                        break
                    op, args = "IMAD.X", "%s, RZ, RZ, RZ, %s" % (x.ops[0], x.ops[3][1:])
                    x.new_op, x.new_args = op, args
                    x.ops = [t.strip() for t in args.split(",")]
                    x.op = op
                    x.compute_rw()
                    touched.add(x.idx)
                    if k > 0:
                        touched.add(b.ins[k - 1].idx)
                    conv += 1
                    continue
                p = parse_iadd3(x)
                if not p:
                    continue
                d, outs, srcs, ins = p
                if outs or len(ins) > 1:
                    continue
                regs = [t for t in srcs if t.is_reg]
                if any(t.t != t.reg for t in regs) or any(t.is_imm for t in srcs):
                    continue
                if len(regs) > 2:
                    continue
                a = regs[0].t if regs else "RZ"
                c = regs[1].t if len(regs) > 1 else "RZ"
                if ins:
                    op, args = "IMAD.X", "%s, %s, 0x1, %s, %s" % (d, a, c, ins[0])
                else:
                    op, args = "IMAD.IADD", "%s, %s, 0x1, %s" % (d, a, c)
                if conv >= limit:
                    break
                x.new_op, x.new_args = op, args
                x.ops = [t.strip() for t in args.split(",")]
                x.op = op
                x.compute_rw()
                touched.add(x.idx)
                if k > 0:
                    touched.add(b.ins[k - 1].idx)
                conv += 1
        print("converted to IMAD:", conv)
    if dry:
        return

    # ---- stall recomputation on the surviving linear sequence
    seq = [x for x in ins_list if not x.deleted]
    for x in seq:
        x.stall = max(x.stall, 1) if x.stall else x.stall
    def recompute():
        bumps = 0
        pos = {}
        cyc = 0
        lastw = {}
        for k, x in enumerate(seq):
            need = 0
            for r in x.rregs + x.rpreds:
                if r in lastw:
                    j, pj = lastw[r]
                    if j.rb != "-" or j.wb != "-":
                        continue   # scoreboard-guarded producer
                    need = max(need, lat(j.op, x.op, "P" if r.startswith("P") else "R") - (cyc - pj))
            if need > 0 and k > 0:
                seq[k - 1].stall += need
                cyc += need
                bumps += 1
            pos[x] = cyc
            for r in x.wregs + x.wpreds:
                lastw[r] = (x, cyc)
            cyc += max(x.stall, 1)
        # loop back-edges: block B targeted by a later block E
        for b in blocks:
            bins = [x for x in b.ins if not x.deleted]
            if not bins:
                continue
            for e in blocks:
                eins = [x for x in e.ins if not x.deleted]
                if not eins or b not in e.succ or seq.index(eins[-1]) < seq.index(bins[0]):
                    continue
                # distance from a write in E (or before E linearly) to a read in B
                end_pos = pos[eins[-1]] + max(eins[-1].stall, 1)
                redefined = set()
                for x in bins:
                    for r in x.rregs + x.rpreds:
                        if r in redefined:
                            continue
                        # last write of r at or before E's end, linearly
                        j = None
                        for y in reversed(seq[:seq.index(eins[-1]) + 1]):
                            if r in y.wregs + y.wpreds:
                                j = y
                                break
                        if j is None or j.rb != "-" or j.wb != "-":
                            continue
                        dist = (end_pos - pos[j]) + (pos[x] - pos[bins[0]])
                        need = lat(j.op, x.op, "P" if r.startswith("P") else "R") - dist
                        if need > 0:
                            eins[-1].stall += need
                            bumps += 1
                            end_pos += need
                    redefined |= set(x.wregs + x.wpreds)
        return bumps
    it = -1
    if "--keep-stalls" not in sys.argv:
        for it in range(20):
            if recompute() == 0:
                break
    print("stall recompute iterations:", it + 1)
    if "--safe-stalls" in sys.argv:
        for k, x in enumerate(seq):
            if hasattr(x, "new_op"):
                x.stall = max(x.stall, 6)
                if k > 0:
                    seq[k - 1].stall = max(seq[k - 1].stall, 6)
                if k > 1:
                    seq[k - 2].stall = max(seq[k - 2].stall, 6)
    over = [x for x in seq if x.stall > 15]
    if over:
        print("WARNING: stall > 15 on", len(over), "instructions; clamped")
        for x in over:
            x.stall = 15

    # ---- emit
    out = []
    cnt = 0
    for i, l in enumerate(lines):
        if start <= i < end:
            x = next((o for k, j, o in items if j == i and k == "ins"), None)
            if x is not None:
                if x.deleted:
                    continue
                head = re.sub(r":S\d\d\]", ":S%02d]" % x.stall, x.head)
                if hasattr(x, "new_op") or i in touched:
                    op, args = (x.new_op, x.new_args) if hasattr(x, "new_op") else (x.op, ", ".join(x.ops))
                    body = "%s%s %s" % (x.gpred or "", op, args) if args else "%s%s" % (x.gpred or "", op)
                    body = body.replace(".reuse", "")
                    l = "%s/*%s*/                   %s ;%s" % (head, x.addr, body, x.tail)
                else:
                    l = head + l[len(x.head):]
                cnt += 1
        out.append(l)
    open(dst, "w").write("\n".join(out))
    print("instructions in kernel after fusion:", cnt)


if __name__ == "__main__":
    main()
