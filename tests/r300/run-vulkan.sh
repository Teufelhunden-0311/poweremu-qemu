#!/bin/sh
# Build and run the Vulkan renderer's interlock test (Linux, against a
# configured QEMU build): ./run-vulkan.sh [builddir] [case name filter]
set -e
T=$(cd "$(dirname "$0")" && pwd)
B=$(cd "${1:-$T/../../build}" && pwd)
O=${TMPDIR:-/tmp}/r300-vk-tests
mkdir -p "$O"

# The flags the build compiles the Vulkan backend with.
CFLAGS=$(python3 - "$B" <<'EOF'
import json, shlex, sys
for c in json.load(open(sys.argv[1] + "/compile_commands.json")):
    if c["file"].endswith("ppc_mac_gpu_vulkan.c"):
        a = shlex.split(c["command"])[1:]
        out, skip = [], False
        for x in a:
            if skip:
                skip = False
            elif x in ("-o", "-c", "-MQ", "-MF"):
                skip = True
            elif x != "-MD":
                out.append(x)
        print(shlex.join(out))
        break
EOF
)
[ -n "$CFLAGS" ] || { echo "no ppc_mac_gpu_vulkan.c in $B/compile_commands.json"; exit 1; }
CC=$(python3 -c "import json,shlex,sys; print([shlex.split(c['command'])[0] for c in json.load(open(sys.argv[1]+'/compile_commands.json')) if c['file'].endswith('ppc_mac_gpu_vulkan.c')][0])" "$B")

R=$T/../../hw/display/r300
cd "$B"
for f in test_vkil:$T/test_vkil.c r300_state:$R/r300_state.c r300_pvs:$R/r300_pvs.c \
         r300_us:$R/r300_us.c r300_spirv:$R/r300_spirv.c r300_draw:$R/r300_draw.c; do
    eval "$CC $CFLAGS -Wno-missing-prototypes -c ${f#*:} -o $O/${f%%:*}.o"
done
# Linked like QEMU's unit tests: QOM and the event loop base, libqemuutil.
LIBS=$(pkg-config --libs gio-2.0 gobject-2.0 gmodule-2.0 glib-2.0 vulkan shaderc)
$CC -o "$O/test_vkil" "$O"/test_vkil.o "$O"/r300_*.o "$B"/libevent-loop-base.a.p/*.o \
    "$B"/libqom.a.p/*.o -Wl,--start-group "$B/libqemuutil.a" $LIBS -lrt -lm -pthread \
    -Wl,--end-group
cd "$T"
"$O/test_vkil" ${2:+"$2"}
