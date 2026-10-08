"""cmp_dis.py NAME KSYM: compare the instruction text in NAME.cuasm (kernel KSYM) with what
nvdisasm makes of the assembled NAME.cubin; prints every mismatching instruction kind once."""
import re
import subprocess
import sys

name, ksym = sys.argv[1], sys.argv[2]
lines = open(name + ".cuasm").read().split("\n")
is_sec = lambda l: re.match(r"\s*\.section\s", l) is not None
start = next(i for i, l in enumerate(lines) if is_sec(l) and ".text." + ksym in l)
end = next(i for i in range(start + 1, len(lines)) if is_sec(lines[i]))
want = []
for l in lines[start:end]:
    m = re.match(r"^\s*\[[^\]]*\]\s*/\*[0-9a-f]+\*/\s*(.*?)\s*;", l)
    if m:
        want.append(re.sub(r"\s+", " ", m[1].replace(".reuse", "")))
out = subprocess.run(["cuobjdump", "-sass", name + ".cubin"], capture_output=True, text=True).stdout
got, on = [], False
for l in out.split("\n"):
    if "Function : " in l:
        on = l.strip().endswith(ksym)
        continue
    if not on:
        continue
    m = re.match(r"^\s*/\*[0-9a-f]{4}\*/\s*(.*?)\s*;", l)
    if m:
        got.append(re.sub(r"\s+", " ", m[1].replace(".reuse", "")))
print("instructions want %d got %d" % (len(want), len(got)))
seen = set()
for i, (w, g) in enumerate(zip(want, got)):
    if w != g and "`" not in w:
        key = re.sub(r"R\d+", "R", re.sub(r"P\d", "P", w))
        if key in seen:
            continue
        seen.add(key)
        print("#%d want: %s\n     got:  %s" % (i, w, g))
