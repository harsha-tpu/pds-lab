#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    int data;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get the rank of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Root process (rank 0) initializes the data
    if (rank == 0) {
        data = 42;
        printf("Root process (rank 0) has data: %d\n", data);
    }
    
    // Broadcast the data from root process (rank 0) to all processes
    MPI_Bcast(&data,              // Address of the data to send/receive
              1,                  // Number of elements to broadcast
              MPI_INT,            // Data type of elements
              0,                  // Rank of the root process
              MPI_COMM_WORLD);    // Communicator
    
    // Each process prints the received data
    printf("Process %d received data: %d\n", rank, data);
    
    // Finalize MPI environment
    MPI_Finalize();
    
    return 0;
}
