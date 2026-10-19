# Doom 3 diagnostics, 2026-10-19 (debug only, not for merging)

AI-generated (Claude Code, Opus 5.5) for review by Codex. Raw data stays on the
author's machine (/Volumes/Claude/d3-ab/*); these are the tools and the notes that
the reported numbers came from.

- d3-classify-2026-10-19.md: the running notes (ablations, profiles, wait attribution,
  CP park 250 vs 50, CP BQL-held breakdown).
- Profilers (this branch): QEMU_TBMAP (accel/tcg/translate-all.c, tb-maint.c) and
  PPC_WAIT_PROF (system/cpus.c, include/system/waitprof.h, hooks in tcg-accel-ops-rr.c,
  openpic.c, ppc_mac_gpu.c, target/ppc/cpu_init.c, qemu-thread-posix.c, runstate.c).
- d3.sh / q32.sh / d3map.sh + run.py / run-map.py: drive demo1 in the VM.
- tbprof.py: host `sample` JIT leaves + tbmap + guest module maps -> guest-code profile.
- wpagg.py: sum PPC_WAIT_PROF 10 s windows inside the demo.
- cpsplit.py / cpincl.py / leafcallers.py: CP thread lock state and BQL-held breakdown.
- gdbgrab.py / gdis.py: guest code via the gdb stub, disassembled with llvm-objdump.
