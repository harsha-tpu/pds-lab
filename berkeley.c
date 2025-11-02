/* Clock Synchronization */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    srand(time(NULL) + rank);
    int local = 100 + (rand() % 20);
    printf("Rank %d: Local time before adjustment %d\n", rank, local);
    int master;
    if (rank == 0) master = local;
    MPI_Bcast(&master, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int diff = local - master;
    int alldiff[size];
    MPI_Gather(&diff, 1, MPI_INT, alldiff, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int sumd;
    MPI_Reduce(&diff, &sumd, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    int avgd, nmaster, myadj;
    int adjust[size];
    if (rank == 0) { 
        avgd = sumd / size;
        nmaster = master + avgd;
        printf("Rank %d: New master time %d\n", rank, nmaster);
        for (int i = 0; i < size; i++) {
            adjust[i] = nmaster - (master + alldiff[i]);
        }
        myadj = adjust[0];
        local += myadj;
        for (int i = 1; i < size; i++) {
            MPI_Send(&adjust[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&myadj, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        local += myadj;
    }
    printf("Rank %d: Local time adjusted to %d\n", rank, local);
    MPI_Finalize();
    return 0;
}
