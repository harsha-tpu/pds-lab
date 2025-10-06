# mutual exclusion -- ensures that only one process enters
the critical section at a time.

# ricart-agrawala algorithm (decentralized algorithm)

- when a process wants to enter the critical section, it sends a 
request message to all other processes, each process replies to the request 
if it is NOT in the critical sectio or if it HAS A 
LOWER PRIORITY REQUEST
- The requesting process enters the critical section only after
receiving replies from all other processes. 


