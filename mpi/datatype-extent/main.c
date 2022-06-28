#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int sendarray[8][6];
    int recvarray[8][6];
    MPI_Datatype vector, vector2;
    MPI_Aint lb, extent;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize arrays
    if (rank == 0) {
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                sendarray[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            recvarray[i][j] = 0;
        }
    }

    if (rank == 0) {
        printf("Data in rank 0\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", sendarray[i][j]);
            }
            printf("\n");
        }
    }

    // TODO create datatype 
    MPI_Datatype coltype, temptype;
    MPI_Type_vector(8, 1, 6, MPI_INT, &coltype);
    MPI_Type_commit(&coltype);
    MPI_Type_get_extent(coltype, &lb, &extent);
    printf("lb=%ld\n", lb);
    printf("extent=%ld\n", extent);
    MPI_Aint addr1, addr2, addrsize;
    MPI_Get_address(&sendarray[0][0], &addr1);
    MPI_Get_address(&sendarray[0][1], &addr2);
    addrsize = addr2-addr1;
    printf("addrsize=%ld\n", addrsize);
    printf("sizeofint=%ld\n", sizeof(int));
    if ( extent != addrsize ) {
        printf("Extent needs to be modified!\n");
        temptype = coltype;
        MPI_Type_create_resized(temptype, 0, sizeof(int), &coltype);
        MPI_Type_commit(&coltype);
        MPI_Type_free(&temptype);
    }
    // Communicate with the datatype
    int tag = 1;
    if (rank == 0) {
        MPI_Send(sendarray, 2, coltype, 1, tag, MPI_COMM_WORLD); 
    } else if (rank == 1) {
        MPI_Recv(recvarray, 2, coltype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
//

    if (rank == 1) {
        printf("Received data\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", recvarray[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Scatter(sendarray, 1, coltype, recvarray, 1, coltype, 0, MPI_COMM_WORLD);
    
    if (rank == 1) {
        printf("Received data\n");
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 6; j++) {
                printf("%3d", recvarray[i][j]);
            }
            printf("\n");
        }
    }

    // free datatype
    MPI_Type_free(&coltype);
    // TODO end
    MPI_Finalize();

    return 0;
}
