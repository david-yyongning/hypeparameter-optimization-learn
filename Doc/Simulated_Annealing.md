
### Simulated Annealing
SA is a probabilistic and metaheuristic optimization technique for approximating the global optimum of a given function.   
* At each iteration, SA selects a neighboring solution s*, and probabistitically decides if moving from current state s to s*.   
* P(e, e*, T) – acceptance probability, energy e = E(s). When T -> 0, P -> 0 if e* > e.
  Early in the process, higher T allows SA to accept worse solutions with a higher probability, enabling exploration of the solution space to avoid local minima. When T decreases, SA becomes more selective, increasingly favoring better solution, ultimately converging towards a global optimum. 
* Kirkpatrick method: P is 1 if e* < e. Otherwise 𝑒xp(−(𝑒∗−𝑒)/𝑇)   
* Temperature (annealing / cooling schedule) schedule T(r): decides how T is reducing during simulation.
* Neighboring function: generate a neighboring solution S* from the current solution S
* Practical issues  
   * Adaptive SA to connect cooling schedule to the search progress or T at each step base on E(s*) – E(s). Ie, optimal cooling rate is critical to balance exploration and exploitation that not universally predetermined.
   * Restart of SA: move back to a solution, eg, s_best if current E(s) is too high
