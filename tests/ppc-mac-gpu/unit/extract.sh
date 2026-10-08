#!/bin/sh
# extract.sh SRC: writes cache.inc (the AGP/GART page cache) and helpers.inc
# (the R300 address helpers), verbatim, from hw/display/ppc_mac_gpu.c
awk '/^static struct \{ uint32_t tag; uint8_t \*host; \} r200_agp_tc/{p=1} p{print} p&&/^static uint8_t \*r200_agp_page\(PPCMacGPUState \*s, uint32_t gpu_addr\)$/{q=1} q&&/^}/{exit}' "$1" > cache.inc
awk '/^static bool r300_span\(/{p=1} p{print} p&&/^static bool r300_sysmem_rw\(/{q=1} q&&/^}/{exit}' "$1" > helpers.inc
