#!/bin/sh
# Build and run the offline R300 tests: ./run.sh [builddir]
set -e
cd "$(dirname "$0")"
B=${1:-${TMPDIR:-/tmp}/r300-tests}
mkdir -p "$B"
R=../../hw/display/r300
SRC="$R/r300_state.c $R/r300_pvs.c $R/r300_us.c $R/r300_draw.c"
for t in test_pvs test_us test_draw test_features; do
    cc -O1 -Wall -Wno-unused-function -o "$B/$t" $t.c $SRC -lm
done
clang -fobjc-arc -framework Metal -framework Foundation mslcheck.m -o "$B/mslcheck"
clang -O1 -fobjc-arc -framework Metal -framework Foundation test_zs.m $SRC -o "$B/test_zs"
clang -O1 -fobjc-arc -framework Metal -framework Foundation test_fmt.m $SRC -o "$B/test_fmt"
clang -O1 -fobjc-arc -framework Metal -framework Foundation test_raster.m $SRC -o "$B/test_raster"
"$B/test_pvs"
"$B/test_draw"
"$B/test_us" > "$B/qe.metal"
"$B/mslcheck" "$B/qe.metal"
"$B/test_features"
"$B/test_zs"
"$B/test_fmt"
"$B/test_raster"
echo "all R300 tests passed"
