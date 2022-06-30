#include <cstdio>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[])
{
    int my_id, ntasks;
    int provided, required=MPI_THREAD_MULTIPLE;

    /* TODO: Initialize MPI with thread support. */
    MPI_Init_thread(&argc, &argv, required, &provided);

    /* TODO: Find out the MPI rank and thread ID of each thread and print
     *       out the results. */
     MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
     MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
     if (my_id == 0) printf("\n");

     #pragma omp parallel
     {
         int omp_rank = omp_get_thread_num();
         int omp_nthreads = omp_get_num_threads();
         // printf("MPI_rank=%d, thread_id=%d\n", my_id, omp_rank);
         int message = omp_nthreads*my_id+omp_rank;

         if (my_id == 0) {
             // All omp tasks of this MPI rank (task) send their data
             for (int i = 1; i<ntasks; ++i) {
                 MPI_Send(&message, 1, MPI_INT, i, omp_rank, MPI_COMM_WORLD);
             }
         } else {
             MPI_Recv(&message, 1, MPI_INT, 0, omp_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         }

         printf("MPI_rank=%d, thread_id=%d, message=%d\n", my_id, omp_rank, message);

     }

    MPI_Finalize();
    return 0;
}
