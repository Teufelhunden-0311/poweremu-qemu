/* ppc-mmu harness: the whole image is the mac99 firmware ROM. */
ENTRY(_start)
SECTIONS
{
    . = 0xfff00000;
    .vectors : { KEEP(*(.vectors)) }
    . = 0xfff08000;              /* past macio-nvram, mapped over 0xfff04000-0xfff07fff */
    .text : { *(.text*) }
    .rodata : { *(.rodata*) *(.srodata*) *(.data.rel.ro*) }
    .data : { *(.data*) *(.sdata*) *(.bss*) *(.sbss*) *(COMMON) }
    /DISCARD/ : { *(.comment) *(.eh_frame*) *(.note*) }
}
