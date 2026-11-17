CFLAGS := -O3 -march=native -pthread -Wall -Wextra

tlm: src/tlm.c
	$(CC) $(CFLAGS) -o $@ $< -lm

# tlm as a static library (API: src/tlm.h); link with -lm -pthread
libtlm.a: src/libtlm.c src/tlm.c src/tlm.h
	$(CC) $(CFLAGS) -c -o libtlm.o src/libtlm.c && ar rcs $@ libtlm.o && rm libtlm.o

lib: libtlm.a

.PHONY: lib
