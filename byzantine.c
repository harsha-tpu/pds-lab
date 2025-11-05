/*
 * Simple MPI program to demonstrate detection of Byzantine faulty nodes.
 * Nodes send their "vote" (rank) to each other. Faulty nodes send random values.
 * Each node collects votes from others and applies majority rule to detect faulty nodes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int faulty = 0;
    if (rank == 2) faulty = 1;
    if (faulty) {
        printf("P%d is faulty, find me!\n", rank);
        fflush(stdout);
    }

    MPI_Status status;

    // --- Stage 1: Each process sends one integer to every other ---
    int recv[size];
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            recv[i] = rank;
            continue;
        }

        int send_val = faulty ? (rand() % 100) : rank;
        MPI_Sendrecv(&send_val, 1, MPI_INT, i, 0,
                     &recv[i], 1, MPI_INT, i, 0,
                     MPI_COMM_WORLD, &status);
    }

    // --- Stage 2: Exchange entire recv[] arrays pairwise ---
    int matrix[size][size];
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            for (int j = 0; j < size; j++)
                matrix[i][j] = recv[j]; // self-copy
            continue;
        }

        MPI_Sendrecv(recv, size, MPI_INT, i, 1,
                     matrix[i], size, MPI_INT, i, 1,
                     MPI_COMM_WORLD, &status);
    }

    // --- Stage 3: Fault detection ---
    for (int i = 0; i < size; i++) {
        int count = 0;
        for (int j = 0; j < size - 1; j++) {
            int votes = 0;
            for (int k = 0; k < size - 1; k++) {
                if (matrix[j][i] == matrix[k][i])
                    votes++;
            }
            if (votes > (size / 2)) {
                count = 1;
                break;
            }
        }
        if (!count) {
            printf("P%d suspects %d is faulty!\n", rank, i);
            fflush(stdout);
        }
    }

    MPI_Finalize();
    return 0;
}
