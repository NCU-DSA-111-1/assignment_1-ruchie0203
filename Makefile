LDFLAGS = -pthread -lpthread
CFLAGS = -g -Wall -Werror
CFILE = $(wildcard src/*.c)
OFILE = $(subst src,build,$(patsubst %.c,%.o,$(CFILE)))
main = $(notdir $(OFILE))
backprop: $(main)
	$(CC) $(LDFLAGS) -o bin/backprop $(OFILE) -lm

main.o: src/main.c
	$(CC) $(CFLAGS) -c src/main.c -o build/main.o

layer.o: src/layer.c
	$(CC) $(CFLAGS) -c src/layer.c -o build/layer.o

neuron.o: src/neuron.c
	$(CC) $(CFLAGS) -c src/neuron.c -o build/neuron.o
nn: bin/backprop
	./bin/backprop
# remove object files and executable when user executes "make clean"
clean:
	rm *.o backprop
