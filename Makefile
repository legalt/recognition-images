PROG = train
CC = g++
CPPFLAGS = -g -Wall -pedantic -std=c++11
OBJS = build/main.o build/bmp.o build/annlib.o

$(PROG): $(OBJS)
	$(CC) -o bin/$(PROG) $(OBJS)

build/main.o:
	$(CC) $(CPPFLAGS) -c sources/main.cpp -o build/main.o

build/bmp.o:
	$(CC) $(CPPFLAGS) -c sources/bmp.cpp -o build/bmp.o

build/annlib.o:
	$(CC) $(CPPFLAGS) -c sources/annlib.cpp -o build/annlib.o

clean:
	rm -rf $(OBJS) $(PROG)
