# mutual exclusion -- ensures that only one process enters
the critical section at a time.

# ricart-agrawala algorithm (decentralized algorithm)

- when a process wants to enter the critical section, it sends a 
request message to all other processes, each process replies to the request 
if it is NOT in the critical sectio or if it HAS A 
LOWER PRIORITY REQUEST
- The requesting process enters the critical section only after
receiving replies from all other processes. 

#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>
#include <mpi.h>

#define REQUEST 0
#define REPLY 1

int main(int argc, char** argv) {
    int rank, size; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    int time = 0; 
    int incs = 0; 
    int want = 0; 
    int need = size - 1; 
    int recv = 0; 
    int queue[100]; 
    int n = 0; 
    srand(rank + 1);
    for (int i = 0; i < 2; i++) {
        sleep(rand() % 3 + 1);
        printf("P%d wants to enter CS\n", rank);
        want = 1; 
        time++; 
        recv = 0;
        
        int req_msg[2] = {time, rank};
        for (int i = 0; i < size; i++) {
            if (i != rank) {
                MPI_Send(req_msg, 2, MPI_INT, i, REQUEST, MPI_COMM_WORLD);
                printf("P%d sent request to %d\n", rank, i);
            }
        }
        
        while (recv < need) {
            MPI_Status status; 
            int msg[2]; 
            MPI_Recv(msg, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            
            if (status.MPI_TAG == REQUEST) {
                printf("P%d got request from %d\n", rank, msg[1]);
                
                if (!incs && !want) {
                    MPI_Send(NULL, 0, MPI_INT, msg[1], REPLY, MPI_COMM_WORLD);
                    printf("P%d replied to %d\n", rank, msg[1]);
                } else if (want && (msg[0] < time || (msg[0] == timestamp && msg[1] < rank))) {
                    MPI_Send(NULL, 0, MPI_INT, msg[1], REPLY, MPI_COMM_WORLD);
                    printf("P%d replied to higher priority %d\n", rank, msg);
                } else {
                    queue[n++] = msg[1];
                    printf("P%d deffered %d (queue: %d)\n", rank, msg[1], n);
                }
                
            } else if (status.MPI_TAG == REPLY) {
                n++; 
                printf("P%d got reply from %d (%d/%d)\n", rank, status.MPI_SOURCE, recv, need);
            }
        }
        
        incs = 1; 
        printf("*** P%d: Entered CS ***\n", rank);
        sleep(2);
        printf("*** P%d: Exited CS ***\n", rank);
        incs = 0; 
        want = 0;
        
        for (int i = 0; i < n; i++) {
            MPI_Send(NULL, 0, MPI_INT, queue[i], REPLY, MPI_COMM_WORLD);
            printf("P%d sent deferred reply to %d\n", rank, queue[i]);
        }
        
        n = 0;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    printf("P%d finished\n", rank);
    
    MPI_Finalize();
    return 0;
}
