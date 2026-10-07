"""sassdf.py: parse a CuAssembler .cuasm kernel, build register/predicate dataflow, report
dependent-pair stall statistics (min issue-cycles between a producer and its consumer per producer
opcode). Shared parsing code for the fusion pass."""
import re
import sys
from collections import defaultdict, Counter

INS_RE = re.compile(r"^(\s*\[(B[-0-9]+):R(.):W(.):(.):S(\d\d)\]\s*)/\*([0-9a-f]+)\*/\s*(@!?P\w\s+)?([A-Z][A-Z0-9._]*)\s*([^;]*);(.*)$")
PAIR_DEST = ("IMAD.WIDE", "LDC.64", "LDG.E.64", "S2UR", "CS2R")
MEM_OPS = ("STG", "ST.", "RED", "ATOMG", "BAR", "MEMBAR", "BSSY", "BSYNC", "NOP", "BRA", "EXIT", "WARPSYNC", "DEPBAR", "ERRBAR")


def reg_of(t):
    t = t.replace(".reuse", "")
    m = re.fullmatch(r"[-~|]*(R\d+|RZ)\|?", t)
    return m[1] if m else None


def pred_of(t):
    m = re.fullmatch(r"!?(P\d|PT)", t)
    return m[1] if m else None


def nxt(r):
    return "R%d" % (int(r[1:]) + 1)


class Ins:
    def __init__(self, line, idx):
        m = INS_RE.match(line)
        self.ok = bool(m)
        if not m:
            return
        self.idx = idx
        self.line = line
        (self.head, self.bar, self.rb, self.wb, self.y, stall, self.addr, self.gpred, self.op, args, self.tail) = m.groups()
        self.stall = int(stall)
        self.ops = [t.strip() for t in args.split(",")] if args.strip() else []
        self.deleted = False
        self.compute_rw()

    def compute_rw(self):
        op, ops = self.op, self.ops
        self.wregs, self.rregs, self.wpreds, self.rpreds = [], [], [], []
        if self.gpred:
            self.rpreds.append(pred_of(self.gpred.strip()))
        if op.startswith(MEM_OPS):
            for t in ops:
                r = reg_of(t)
                p = pred_of(t)
                if r and r != "RZ":
                    self.rregs.append(r)
                if p and p != "PT":
                    self.rpreds.append(p)
                m = re.search(r"\[(R\d+)", t)
                if m:
                    self.rregs += [m[1], nxt(m[1])]
                if "64" in op and r and r != "RZ":
                    self.rregs.append(nxt(r))
            if op.startswith("ATOMG"):
                d = reg_of(ops[0])
                if d and d != "RZ":
                    self.wregs.append(d)
                    self.rregs = [r for r in self.rregs if r != d]
            return
        if op.startswith(("ISETP", "PLOP3", "P2R", "R2P")):
            k = 0
            if op.startswith(("ISETP", "PLOP3")):
                self.wpreds = [pred_of(ops[0]), pred_of(ops[1])]
                k = 2
            for t in ops[k:]:
                r = reg_of(t)
                p = pred_of(t)
                if r and r != "RZ":
                    self.rregs.append(r)
                if p and p != "PT":
                    self.rpreds.append(p)
            if op.startswith("P2R"):
                self.wregs.append(reg_of(ops[0]))
                self.rregs = []
            if op.startswith("R2P"):
                self.rregs = [reg_of(ops[1])]
            return
        if not ops:
            return
        d = reg_of(ops[0])
        if d is None:
            # uniform-datapath or other ops: track plain register reads only
            for t in ops:
                r = reg_of(t)
                p = pred_of(t)
                if r and r != "RZ":
                    self.rregs.append(r)
                if p and p != "PT":
                    self.rpreds.append(p)
            return
        if d != "RZ":
            self.wregs.append(d)
            if op.startswith(PAIR_DEST):
                self.wregs.append(nxt(d))
        k = 1
        while k < len(ops) and pred_of(ops[k]) and not ops[k].startswith("!") and ops[k] != "PT":
            self.wpreds.append(pred_of(ops[k]))
            k += 1
        srcs = ops[k:]
        npred_tail = sum(1 for t in srcs if pred_of(t))
        nreg = len(srcs) - npred_tail
        for j, t in enumerate(srcs):
            r = reg_of(t)
            p = pred_of(t)
            if r and r != "RZ":
                self.rregs.append(r)
                if op.startswith("IMAD.WIDE") and j == nreg - 1 and j >= 2:
                    self.rregs.append(nxt(r))
            if p and p != "PT":
                self.rpreds.append(p)
            m = re.search(r"c\[0x[0-9a-f]+\]\[(R\d+)", t)
            if m:
                self.rregs.append(m[1])
        if op.startswith("SEL") and self.wpreds:
            self.rpreds += self.wpreds
            self.wpreds = []


def load(path):
    lines = open(path).read().split("\n")
    ins = []
    for i, l in enumerate(lines):
        x = Ins(l, i)
        if x.ok:
            ins.append(x)
    return lines, ins


def opclass(op):
    return op.split(".")[0] + (".WIDE" if op.startswith("IMAD.WIDE") else "")


def latency_table(ins, by_consumer=False):
    """min issue-cycle distance producer->consumer observed in the schedule, per producer class
    (and consumer class when by_consumer); scoreboard-guarded producers are skipped"""
    lat = defaultdict(lambda: 99)
    lastw = {}
    cyc = 0
    for x in ins:
        for r in x.rregs + x.rpreds:
            if r in lastw:
                p, pc = lastw[r]
                if p.rb != "-" or p.wb != "-":
                    continue
                kind = "P" if r.startswith("P") else "R"
                key = (opclass(p.op), opclass(x.op), kind) if by_consumer else (opclass(p.op), kind)
                lat[key] = min(lat[key], cyc - pc)
        for r in x.wregs + x.wpreds:
            lastw[r] = (x, cyc)
        cyc += max(x.stall, 1)
    return lat


if __name__ == "__main__":
    lines, ins = load(sys.argv[1])
    print("instructions:", len(ins))
    lat = latency_table(ins, by_consumer="--by-consumer" in sys.argv)
    for k in sorted(lat):
        print("min issue-cycles producer->consumer", k, lat[k])
    print("stall histogram:", sorted(Counter(x.stall for x in ins).items()))
    print("yield count:", sum(1 for x in ins if x.y == "Y"), "barriers (R/W set):", sum(1 for x in ins if x.rb != "-" or x.wb != "-"))
