#ifndef VKMOCK_QEMU_BSWAP_H
#define VKMOCK_QEMU_BSWAP_H
#ifndef bswap32
#define bswap32(x) __builtin_bswap32(x)
#endif
#endif
