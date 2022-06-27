#include <cstdio>
#include <cmath>
#include <mpi.h>

constexpr int n = 840;

int main(int argc, char** argv)
{

  printf("Computing approximation to pi with N=%d\n", n);
  
  int myid, ntasks;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  

  // int istart = (myid==0)?1:n/2+1;
  // int istop = (myid==0)?n/2:n;
  
  int istart = myid*(n/ntasks)+ 1 + (n%ntasks)>0?myid%(n%ntasks):0;
  int istop  = (myid+1)*(n/ntasks)  + (n%ntasks)>0?myid%(n%ntasks):0;
   

  printf("rank %i: start=%i, stop=%i\n", myid, istart, istop);

  double pi = 0.0;
  for (int i=istart; i <= istop; i++) {
    double x = (i - 0.5) / n;
    pi += 1.0 / (1.0 + x*x);
  }
  
  int tag = 1, msgsize = 1;
  if (myid == 0) {
    double other_part[1];
    for (int i = 1; i < ntasks ; ++i) {
        MPI_Recv(other_part, msgsize, MPI_DOUBLE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        pi += other_part[0]; 
    }
    pi *= 4.0 / n;
    printf("Approximate pi=%18.16f (exact pi=%10.8f)\n", pi, M_PI);
  } else {
    double send_arr[] = {pi};
    MPI_Send(send_arr, msgsize, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;

}
   

