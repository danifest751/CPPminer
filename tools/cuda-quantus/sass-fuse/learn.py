import sys
sys.path.insert(0, "/w/bench/CuAssembler")
from CuAsm.CuInsFeeder import CuInsFeeder
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos
path = "/w/bench/CuAssembler/CuAsm/InsAsmRepos/DefaultInsAsmRepos.sm_75.txt"
r = CuInsAssemblerRepos(path, arch="sm_75")
for f in sys.argv[1:]:
    r.update(CuInsFeeder(f, archfilter="sm_75"))
r.save2file(path)
