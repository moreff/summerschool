#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, omp_rank;
    int provided, required=MPI_THREAD_FUNNELED;

    /* TODO: Initialize MPI with thread support. */
    MPI_Init_thread(&argc, &argv, required, &provided);

    /* TODO: Find out the MPI rank and thread ID of each thread and print
     *       out the results. */
     MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
     if (my_id == 0) printf("\n");
     #pragma omp parallel
     {
         omp_rank = omp_get_thread_num();
         printf("MPI_rank=%d, thread_id=%d\n", my_id, omp_rank);
     }


    /* TODO: Investigate the provided thread support level. */
    if (my_id == 0) {
        printf("Thread support level: %d", provided);
    }

    MPI_Finalize();
    return 0;
}
