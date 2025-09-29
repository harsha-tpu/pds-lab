#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int token;              // carries the max rank seen in election
    int elected_leader;
    int initiator = 1;      // Simulate: Process 1 detects failure and starts election
    int failed_leader;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    failed_leader = size - 1;  // Assume highest rank was leader and failed

    if (rank == initiator) {
        printf("Process %d Detected failure of coordinator process %d and initiated an election.\n", rank, failed_leader);
        fflush(stdout);

        // Start election: send own rank as initial max
        token = rank;
        int next = (rank + 1) % size;
        MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);

        // Wait for token to come back
        int prev = (rank - 1 + size) % size;
        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // I am initiator: token now holds max rank in ring (excluding failed leader)
        elected_leader = token;
        printf("Process %d: Election completed. New coordinator is process %d\n", rank, elected_leader);
        fflush(stdout);

        // Broadcast the new leader to everyone (including self)
        int i;
        for ( i = 0; i < size; i++) {
            if (i != rank) {
                MPI_Send(&elected_leader, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        // Forward the election message until it comes back to initiator
        int prev = (rank - 1 + size) % size;
        int next = (rank + 1) % size;

        MPI_Recv(&token, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Update max rank (if self is higher, update; but skip failed leader)
        if (rank != failed_leader && rank > token) {
            token = rank;
        }

        // If not back to initiator, forward it
        if (next != initiator) {
            MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        } else {
            // Send back to initiator
            MPI_Send(&token, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        }

        // Wait to receive final elected leader from initiator
        MPI_Recv(&elected_leader, 1, MPI_INT, initiator, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // All processes print the result
    printf("Process %d: Coordinator is process %d\n", rank, elected_leader);
    fflush(stdout);

    MPI_Finalize();
    return 0;
}
