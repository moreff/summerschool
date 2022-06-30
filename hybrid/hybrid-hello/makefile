main: main.cpp
	mpicxx -o main main.cpp -fopenmp

debug: main.cpp
	mpicxx -g -O0 main.cpp -o main -fopenmp

sanitize: main.cpp
	mpicxx -O0 main.cpp -o main -fsanitize=address
