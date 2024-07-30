E3NN C++ Enzyme
===============
Idea, C++ (eigen) based implementation of E3NN, with gradients computed by Enzyme.

Basic project structure and implementation order. This project aims to provide incremental progress while minimizing dependency-related interruptions.

Project structure:
```
e3nn/
├── include/
│   ├── o3/
│   ├── nn/
│   ├── io/
│   ├── math/
│   └── util/
├── src/
│   ├── o3/
│   ├── nn/
│   ├── io/
│   ├── math/
│   └── util/
├── tests/
│   ├── o3/
│   ├── nn/
│   ├── io/
│   ├── math/
│   └── util/
├── examples/
└── CMakeLists.txt
```

Implementation order:

1. Start with the `util` namespace:
   - Implement basic utility functions and classes
   - This will provide a foundation for other parts of the library

2. Move on to the `math` namespace:
   - Implement linear algebra functions
   - Add other mathematical utilities

3. Implement the `o3` namespace:
   - Begin with basic classes like `Irrep` and `Irreps`
   - Implement rotation-related functions
   - Add spherical harmonics and Wigner D-matrices
   - Implement tensor product operations

4. Work on the `nn` namespace:
   - Start with basic neural network modules
   - Implement more complex modules that depend on `o3` functionality

5. Finally, implement the `io` namespace:
   - Add functions for handling Cartesian and spherical tensors

For each namespace and major component:
1. Implement the core functionality
2. Write unit tests
3. Create simple examples to demonstrate usage

TODO:
- [ ] Forward functions
- [ ] Backward Enzyme gradients

Contribution
============

If you want to contribute, you can start by porting any function from e3nn pytorch, and putting it in appropriate location
