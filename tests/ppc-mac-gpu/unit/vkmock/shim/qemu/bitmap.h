#ifndef VKMOCK_QEMU_BITMAP_H
#define VKMOCK_QEMU_BITMAP_H
#define BITS_PER_LONG (sizeof(unsigned long) * 8)

static inline unsigned long *bitmap_new(long nbits)
{
    return g_malloc0(DIV_ROUND_UP(nbits, BITS_PER_LONG) * sizeof(unsigned long));
}

static inline void bitmap_zero(unsigned long *dst, long nbits)
{
    memset(dst, 0, DIV_ROUND_UP(nbits, BITS_PER_LONG) * sizeof(unsigned long));
}

static inline unsigned long find_next_bit(const unsigned long *addr,
                                          unsigned long size, unsigned long offset)
{
    for (; offset < size; offset++) {
        if (addr[offset / BITS_PER_LONG] & (1ul << (offset % BITS_PER_LONG))) {
            return offset;
        }
    }
    return size;
}

static inline unsigned long find_first_bit(const unsigned long *addr, unsigned long size)
{
    return find_next_bit(addr, size, 0);
}
#endif
