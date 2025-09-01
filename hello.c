#include <stdio> 
#include <mpi.h> 

void main(int argc, char* argv[]) {
  int myRank; 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  printf("Hello from node %d\n", myRank);
  MPI_Finalize();
}
