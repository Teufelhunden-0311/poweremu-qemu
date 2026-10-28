/*
 * ppc-mmu harness: bare-metal start-up, exception vectors and helpers for
 * qemu-system-ppc -M mac99 -cpu 7400 -bios mmu.elf.  The image is the
 * firmware ROM at 0xfff00000; the CPU starts at 0xfff00100 with MSR[IP]
 * set, so the vectors are here too.  A fault the test expects returns to a
 * recovery address the helper left at FAULT+20.  AI-generated (Claude).
 */
#define STACK   0x0000f000
#define FAULT   0x00003f00      /* vector, DAR, DSISR, SRR0, SRR1, recovery */

    .section .vectors, "ax"
    .globl _vectors
_vectors:
    .org 0x100                  /* system reset */
    .globl _start
_start:
    li      0, 0
    lis     1, STACK@h
    ori     1, 1, STACK@l
    stwu    0, -16(1)
    li      3, 0x1040           /* ME | IP */
    mtmsr   3
    isync
    bl      main
1:  b       1b

#define VEC(off, n)              \
    .org off;                    \
    mtsprg  0, 3;                \
    mtsprg  1, 4;                \
    li      4, n;                \
    b       fault_common

    VEC(0x200, 0x200)           /* machine check */
    VEC(0x300, 0x300)           /* DSI */
    VEC(0x400, 0x400)           /* ISI */
    VEC(0x500, 0x500)
    VEC(0x600, 0x600)
    VEC(0x700, 0x700)           /* program */
    VEC(0x800, 0x800)
    VEC(0x900, 0x900)           /* decrementer (EE is never set) */
    VEC(0xc00, 0xc00)

    .org 0x1000
fault_common:
    li      3, FAULT
    stw     4, 0(3)
    mfdar   4
    stw     4, 4(3)
    mfdsisr 4
    stw     4, 8(3)
    mfsrr0  4
    stw     4, 12(3)
    mfsrr1  4
    stw     4, 16(3)
    lwz     4, 20(3)            /* recovery address, one shot */
    cmpwi   4, 0
    beq     2f
    mtsrr0  4
    li      4, 0
    stw     4, 20(3)
    mfsprg  4, 1
    mfsprg  3, 0
    rfi
2:  li      4, 0x7ead           /* unexpected: stop, leave the record */
    stw     4, 24(3)
3:  b       3b

    .text
/* uint32_t try_store(uint32_t ea, uint32_t val): 0 stored, 1 faulted */
    .globl try_store
try_store:
    lis     5, 1f@h
    ori     5, 5, 1f@l
    li      6, FAULT
    stw     5, 20(6)
    sync
    stw     4, 0(3)
    li      5, 0
    stw     5, 20(6)
    li      3, 0
    blr
1:  li      3, 1
    blr

/* uint32_t try_load(uint32_t ea, uint32_t *out): 0 loaded, 1 faulted */
    .globl try_load
try_load:
    lis     5, 1f@h
    ori     5, 5, 1f@l
    li      6, FAULT
    stw     5, 20(6)
    sync
    lwz     7, 0(3)
    li      5, 0
    stw     5, 20(6)
    stw     7, 0(4)
    li      3, 0
    blr
1:  li      3, 1
    blr

/*
 * uint32_t call_ctr(uint32_t ea): bctrl to translated code, its r3; an ISI
 * comes back as 0xfa000000 | vector.
 */
    .globl call_ctr
call_ctr:
    mflr    0
    stwu    1, -16(1)
    stw     0, 20(1)
    lis     5, 1f@h
    ori     5, 5, 1f@l
    li      6, FAULT
    stw     5, 20(6)
    mtctr   3
    bctrl
    li      5, 0
    li      6, FAULT
    stw     5, 20(6)
    b       2f
1:  li      6, FAULT
    lwz     3, 0(6)
    oris    3, 3, 0xfa00
2:  lwz     0, 20(1)
    addi    1, 1, 16
    mtlr    0
    blr

/*
 * uint32_t call_lr(uint32_t ea): enter translated code by blr (the QEMU
 * bclr lookup path).  The target returns with "mtlr 4; blr", r4 being the
 * return point set here.
 */
    .globl call_lr
call_lr:
    mflr    0
    stwu    1, -16(1)
    stw     0, 20(1)
    lis     5, 1f@h
    ori     5, 5, 1f@l
    li      6, FAULT
    stw     5, 20(6)
    lis     4, 3f@h
    ori     4, 4, 3f@l
    mtlr    3
    blr
3:  li      5, 0
    li      6, FAULT
    stw     5, 20(6)
    b       2f
1:  li      6, FAULT
    lwz     3, 0(6)
    oris    3, 3, 0xfa00
2:  lwz     0, 20(1)
    addi    1, 1, 16
    mtlr    0
    blr
