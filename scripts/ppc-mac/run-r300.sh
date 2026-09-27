#!/bin/sh
# Boot vm/tiger.qcow2 with the emulated ATI Radeon 9700 PRO (R300, 1002:4E44)
# in place of VGA.  Same machine as run-tiger.sh otherwise.
#
# Run it through the vm/run-r300.sh symlink: paths (fw-r300/, the disk)
# are resolved relative to the invoking path, i.e. the vm/ directory.
#
#   firmware  vm/fw-r300/: UTM's OpenBIOS with its QEMU-VGA PCI table entry
#             (1234:1111 at 0x30678) changed to 1002:4E44, so the firmware
#             builds the display node and attaches the NDRV to the Radeon.
#   NDRV      qemu_vga_hwc.ndrv (QEMU VGA NDRV + hardware cursor) drives the
#             framebuffer; Tiger's ATIRadeon9700.kext attaches for 2D/3D.
#   ROM       the Mac 9700 PRO ROM is the PCI expansion ROM and is also
#             copied to VRAM offset 0, where POST would shadow it.
#
# Switches: VERBOSE=off (grey Apple instead of -v), TRACE_GPU=on (every GPU
# register access, to vm/gpu-trace.log; large and slow), AGPBRIDGE=off.
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
QEMU="${QEMU:-$ROOT/qemu/build/qemu-system-ppc}"
FW="$HERE/fw-r300"
DISK="${DISK:-$HERE/tiger.qcow2}"
VRAM_MB=128

export QEMU_PPC_NDRV="$FW/qemu_vga_hwc.ndrv"

# Tell Mac OS X how much VRAM the card has, as the ATI FCode would, and give
# the display node the 9700 PRO FCode's identity (values from the ROM).
# System Profiler maps model "ATY,R300" to "ATI Radeon 9700 Pro" and shows
# ATY,Rom# as the ROM revision.  name/compatible stay "QEMU,VGA": the NDRV
# is matched by them, and neither the ATI kext nor its bundles read these.
ATY_PROPS="\" ATY,R300\" encode-string \" model\" property \" 113-A06500-124\" encode-string \" ATY,Rom#\" property \" 109-A06501-00\" encode-string \" ATY,Card#\" property \" 1.88\" encode-string \" ATY,Fcode\" property"
BOOT_CMD="boot-command=\" /pci@f2000000/QEMU,VGA@e\" ['] find-device catch 0= if h# $(printf %x $((VRAM_MB * 1048576))) encode-int \" VRAM,totalsize\" property $ATY_PROPS device-end then init-program go"

set --  "$@"
[ "${VERBOSE:-on}" = on ] && set -- -prom-env 'boot-args=-v' "$@"
[ "${AGPBRIDGE:-on}" = on ] && set -- -global uni-north-pci.agp-capable=on "$@"
if [ "${TRACE_GPU:-off}" = on ]; then
    set -- -trace 'ppc_mac_gpu_*' "$@"
else
    set -- -trace ppc_mac_gpu_realize "$@"
fi

exec "$QEMU" -L "$FW" -L "$ROOT/qemu/pc-bios" -nodefaults -vga none \
    -machine mac99,via=pmu -accel tcg,tb-size=256 \
    -smp cpus=1,sockets=1,cores=1,threads=1 -m 1024 \
    -device loader,addr=0x4000000,file="$FW/ppc-ndrvloader" \
    -prom-env "$BOOT_CMD" \
    -device ati-radeon-9700,vgamem_mb=$VRAM_MB,romfile="$FW/ati_oem_9700pro_124_agp.rom",biosrom="$FW/ati_oem_9700pro_124_agp.rom" \
    -g 1024x768x32 \
    -display cocoa -audio coreaudio \
    -netdev user,id=net0,ipv6=off,hostfwd=tcp:127.0.0.1:2222-:22 -device sungem,netdev=net0 \
    -usb -device usb-mouse,bus=usb-bus.0 -device usb-kbd,bus=usb-bus.0 \
    -device usb-tablet,bus=usb-bus.0 \
    -drive if=none,media=disk,id=hd,file.filename="$DISK",discard=unmap,detect-zeroes=unmap \
    -device ide-hd,bus=ide.0,unit=0,drive=hd,bootindex=0 \
    -monitor tcp:127.0.0.1:4444,server,nowait \
    -D "$HERE/gpu-trace.log" \
    "$@"
