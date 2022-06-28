#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int n=1000, cnt=3, reps=10000;
  int casei = -1;

  typedef struct {
    float coords[3];
    int charge;
    char label[2];
  } particle;

  particle particles[n];

  int i, j, myid, ntasks;
  double t1, t2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (myid == 0) {
    printf("Enter case number: ");
    fflush( stdout );
    scanf("%d", &casei);
  }

  MPI_Bcast(&casei, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // MPI_Barrier(MPI_COMM_WORLD);
  /* fill in some values for the particles */
  if (myid == 0) {
    for (i=0; i < n; i++) {
      for (j=0; j < 3; j++)
        particles[i].coords[j] = (float)rand()/(float)RAND_MAX*10.0;
      particles[i].charge = 54;
      strcpy(particles[i].label, "H");
    }
  }

  // TODO: define data type for the struct
  MPI_Datatype particletype, temptype, types[cnt]={MPI_FLOAT,MPI_INT,MPI_CHAR};
  MPI_Aint disp[cnt], dist[2], lb, extent;
  int blocklen[cnt] = {3,1,2};
  MPI_Get_address(&particles[0].coords, &disp[0]);
  MPI_Get_address(&particles[0].charge, &disp[1]);
  MPI_Get_address(&particles[0].label, &disp[2]);
  disp[2] -= disp[0];
  disp[1] -= disp[0];
  disp[0] = 0;
  printf("Creating type struct\n"); 
  MPI_Type_create_struct(cnt, blocklen, disp, types, &particletype);
  MPI_Type_commit(&particletype);
  MPI_Type_get_extent(particletype, &lb, &extent);
//printf("Extent = %ld\n", extent);
MPI_Aint addr1, addr2;
MPI_Get_address(&particles[0], &addr1);
MPI_Get_address(&particles[1], &addr2);
int structsize = addr2-addr1;
//printf("Addr dif = %ld\n", addr2-addr1);
//printf("Sizeof = %ld\n", sizeof(particles[0]));
  // TODO: check extent (not really necessary on most platforms) 
  if ( extent != sizeof(particles[0]) ) {
    printf("Warning! Changing the extent\n");
    temptype = particletype;
    MPI_Type_create_resized(temptype, 0, sizeof(particles[0]), &particletype);
    MPI_Type_commit(&particletype);
    MPI_Type_free(&temptype);
  }
  int tag = 1;
  // communicate using the created particletype
  t1 = MPI_Wtime();
  if (casei == 1) {
  if (myid == 0) {
    printf("Sending...\n");
    for (i=0; i < reps; i++) { // multiple sends for better timing
      // TODO: send  
      MPI_Send(particles, n, particletype, 1, i, MPI_COMM_WORLD);
    }
  } else if (myid == 1) {
    printf("Receiving...\n");
    for (i=0; i < reps; i++) {
      // TODO: receive
      MPI_Recv(particles, n, particletype, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  } else if (casei == 2) {
    if (myid == 0) {
      for (i=0; i < reps; i++) { 
      //printf("Sending %d...\n", i);
              MPI_Send(particles, n*structsize, MPI_BYTE, 1, i, MPI_COMM_WORLD);
      } 
    } else if (myid == 1) {
      for (i=0; i < reps; i++) {
      //printf("Receiving %d...\n", i);
        MPI_Recv(particles, n*structsize, MPI_BYTE, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  }

  // TODOs end

  t2 = MPI_Wtime();

  printf("Time: %i, %e \n", myid, (t2-t1)/(double)reps);
  printf("Check: %i: %s %f %f %f \n", myid, particles[n-1].label,
          particles[n-1].coords[0], particles[n-1].coords[1],
          particles[n-1].coords[2]);
  MPI_Type_free(&particletype);
  MPI_Finalize();
  return 0;
}
