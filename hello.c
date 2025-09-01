#include <stdio.h> 
#include <mpi.h> 

void main(int argc, char** argv) {
  int rank, size; 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("Hello from node %d\n", rank);
  printf("No of processors: %d\n", size);
  MPI_Finalize();
}


//---------------------OUTPUT-----------------------

[cse7e23@sastra-masternode ~]$ mpicc hello.c -o out
[cse7e23@sastra-masternode ~]$ mpirun -np 4 out
Hello from node 0
No of processors: 4
Hello from node 1
No of processors: 4
Hello from node 3
No of processors: 4
Hello from node 2
No of processors: 4
