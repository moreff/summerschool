#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank;
    int array[8][8];
    //TODO: Declare a variable storing the MPI datatype

    int i, j;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int casei = 3;
    // Initialize arrays
    if (rank == 0) {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = (i + 1) * 10 + j + 1;
            }
        }
    } else {
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                array[i][j] = 0;
            }
        }
    }

    if (rank == 0) {
        printf("Data in rank 0\n");
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    //TODO: Create datatype 
    if (casei == 1){
        MPI_Datatype newtype;
        MPI_Type_vector(8, 1, 8, MPI_INT, &newtype);
        MPI_Type_commit(&newtype);
        int tag = 11;
        if (rank == 0) {
            MPI_Send(&array[0][1], 1, newtype, 1, tag, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(&array[0][1], 1, newtype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&newtype);
    } else if (casei == 2) {
        MPI_Datatype newtype;
        int blocklens[] = {1,2,3,4};
        int displs[] = {0, 17, 34, 51};
        MPI_Type_indexed(4, blocklens, displs, MPI_INT, &newtype);
        MPI_Type_commit(&newtype);
        int tag = 11;
        if (rank == 0) {
            MPI_Send(array, 1, newtype, 1, tag, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(array, 1, newtype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&newtype);
        
    } else if (casei == 3) {
        MPI_Datatype newtype;
        int sizes[] = {8,8};
        int subsizes[] = {4,4};
        int offsets[] = {2,2};
        MPI_Type_create_subarray(2, sizes, subsizes, offsets, MPI_ORDER_C, MPI_INT, &newtype);
        MPI_Type_commit(&newtype);
        int tag = 11;
        if (rank == 0) {
            MPI_Send(array, 1, newtype, 1, tag, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(array, 1, newtype, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&newtype);
        
    }
    //TODO: Send data

    //TODO: Free datatype

    // Print out the result on rank 1
    if (rank == 1) {
        printf("Received data\n");
        for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
                printf("%3d", array[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();

    return 0;
}
