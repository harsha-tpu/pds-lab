/*
 * Simple MPI program to demonstrate detection of Byzantine faulty nodes.
 * Nodes send their "vote" (rank) to each other. Faulty nodes send random values.
 * Each node collects votes from others and applies majority rule to detect faulty nodes.
 */
#include <stdio.h> 
#include <stdlib.h> 
#include <stdbool.h>
#include <mpi.h> 

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, rank);
  MPI_Comm_size(MPI_COMM_WORLD, size);
  bool isFaulty = false;
  if (rank == 2 || rank == 5) {
    isFaulty = true;
    printf("P%d: I am a faulty node!\n", rank);
  }
  // each process hold votes of every process
  int votes[size];

  // sending their votes
  if (!isFaulty) { // good nodes
    for (int i = 0; i < size; i++) {
      if (i! = rank) MPI_Send(&rank, 1, MPI_INT, i, 101, MPI_COMM_WORLD);
    }
    // receive votes from all other nodes
    for (int i = 0; i < size; i++) {
      if (i == rank) {
        votes[i] = rank;
      } else {
        MPI_Status status; 
        MPI_Recv(&votes[i], 1, MPI_INT, i, 101, MPI_COMM_WORLD, &status);
      }
    }
  } else {
    int randomValue;
    for (int i = 0; i < size; i++) {
      randomValue = (rand() + rank * rank) % 100; 
      MPI_Send(&randomValue, 1, MPI_INT, i, 101, MPI_COMM_WORLD);
    }
  }
  
  MPI_Finalize();
  return 0; 
}

