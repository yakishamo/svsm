.PHONY: all clean

all: init.bin

init.bin: init.asm link.ld
	nasm -f elf64 init.asm -o init.o
	ld -T link.ld init.o -o init.bin

clean:
	rm -f user_hlt user_hlt.o
