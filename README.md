# yeet-pythtb

This package is a modification of the original [PythTB](https://www.physics.rutgers.edu/pythtb/), developed and mantained by by Sinisa Coh (University of California at Riverside), David Vanderbilt (Rutgers University) and [others](https://www.physics.rutgers.edu/pythtb/about.html#history). All credit of the essential implementation goes to the original developers.

This modification implements the most performance-critical routines in a JIT-compiled way by using [Numba](http://numba.pydata.org/). In particular, the functions that solve the Hamiltonian are compiled and parallelized. This affects essential computations such as the calculation of bandstructures and Wannier charge centers, etc.

The package succesfully runs all the [examples](https://www.physics.rutgers.edu/pythtb/examples.html) for the original PythTB, so it is expected to work without issues.
