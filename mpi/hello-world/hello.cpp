#include <cstdio>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int ntasks, myrank, namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks); // getting number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Get_processor_name(procname, &namelen);
    if (myrank == 0) {
        printf("Total number of processes is %d\n", ntasks);
    }
    printf("Hello world from process %d running on %s!\n", myrank, procname);
    MPI_Finalize();
    return 0;
}
