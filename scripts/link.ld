OUTPUT_FORMAT(elf64-x86-64)
ENTRY(_start)

SECTIONS
{
    .text 0x0: {        /* コードセクションをアドレス0から開始（相対アドレス）*/
        *(.text)
    }
    .data : {           /* データセクション */
        *(.data)
    }
    .rodata : {         /* 読み取り専用データセクション */
        *(.rodata)
    }
    .bss : {            /* 初期化されていないデータセクション */
        *(.bss)
    }
}
