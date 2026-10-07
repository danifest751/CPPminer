#!/bin/bash
# hackbuild.sh OUT CUBIN_OR_- nvcc-args...
# Builds $SRC like nvcc would, but replaces the sm_75 cubin produced by ptxas with CUBIN
# ("-" keeps the original and also saves it as OUT.orig.cubin). Runs inside the CUDA container.
set -e
OUT=$1; CUBIN=$2; shift 2; SRC=${SRC:-qb.cu}; BASE=${SRC%.cu}
K=k_$OUT
rm -rf $K; mkdir -p $K
nvcc "$@" -o $OUT $SRC -keep -keep-dir $K --dryrun 2>&1 | sed -n 's/^#\$ //p' > $K/steps.txt
# environment lines become exports; split at the ptxas step
awk -v K=$K '
  /^[A-Za-z_]+=/ { sub(/=/, "=\""); print "export " $0 "\""; next }
  { sub(/^rm /, "rm -f "); print > (done ? K "/post.sh" : K "/pre.sh") }
  /^ptxas / { done = 1 }
' $K/steps.txt > $K/env.sh
( . $K/env.sh; sh -e $K/pre.sh )
if [ "$CUBIN" = "-" ]; then
  cp $K/$BASE.sm_75.cubin $OUT.orig.cubin
else
  cp "$CUBIN" $K/$BASE.sm_75.cubin
fi
( . $K/env.sh; sh -e $K/post.sh )
echo "built $OUT"
