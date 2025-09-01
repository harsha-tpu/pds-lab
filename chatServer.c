#include <stdio.h> 
#include <mpi.h> 

void main() {
  int myRank; 
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank); 
  //
  MPI_Finalize();
}
