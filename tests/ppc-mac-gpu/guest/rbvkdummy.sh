#!/bin/zsh
# rbvkdummy.sh: Vulkan with the Khronos validation layer; do not submit the batch that
# carries the stand-in textures' initialization (test hook), check that they are
# initialized again.  VD-nofix is the negative control (the fix switched off).
# The layer manifest in $VKL must name the layer library by full path.
W=~/ppcosxkvm-work/rbprobe; VKL=${VKL:-$HOME/ppcosxkvm-work/vklayer}
L="VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_PATH=$VKL VK_LOADER_DEBUG=error"
run() { tag=$1; shift; $W/rbvm.sh $tag ${=L} "$@" -- --gpu vulkan | grep FAILED
  O=$W/vm-$tag; echo "== $tag: $(grep -a 'strip:' $O/read.txt)"
  grep -a RBTEST ~/ppcosxkvm/vm/gpu-trace.log > $O/rbtest.txt; sed 's/^/   /' $O/rbtest.txt
  grep -a -E 'submitting to the GPU failed' $O/run.out | sort | uniq -c | sed 's/^/   /'
  echo "   layer failed to load: $(grep -a -c 'failed to load' $O/run.out); validation errors: $(grep -a -c 'Validation Error' $O/run.out)"
  grep -a -A1 'Validation Error' $O/run.out | grep -a -o 'expects VkImage.*' | sed -E 's/0x[0-9a-f]+/H/g' | sort | uniq -c | sed 's/^/     /'; }
run VD-base
run VD-host1 R300_VK_FAIL=dummy:-1:1
run VD-dev3 R300_VK_FAIL=dummy:-2:3
run VD-nofix R300_VK_FAIL=dummy:-1:1 RBTEST_NO_DUMMY_FIX=1
echo VD-DONE
