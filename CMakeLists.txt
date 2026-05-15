cmake_minimum_required(VERSION 3.1)

# Project name
project(SU3_QuantumTrajectories CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the directories for PETSc headers and libraries
include_directories(/home/alberto/petsc_complex/include)
link_directories(/home/alberto/petsc_complex/lib)

# Add the directories for SLEPc headers and libraries
include_directories(/home/alberto/slepc_complex/include)
link_directories(/home/alberto/slepc_complex/lib)

# Sources
set(SOURCES
    ./main.cpp
    ./modules/basis.cpp
    ./modules/tensored_basis.cpp
    ./modules/quantum_trajectory.cpp
    ./modules/effective_hamiltonian.cpp
)

# Add the executable
add_executable(SU3_QuantumTrajectories ${SOURCES})

# Add compile options
target_compile_options(SU3_QuantumTrajectories PRIVATE -Wall -O3 -fconcepts -fopenmp -mavx)

# Add runtime library path
set_target_properties(SU3_QuantumTrajectories PROPERTIES
    INSTALL_RPATH "/usr/lib/x86_64-linux-gnu/pmix/lib"
    BUILD_RPATH "/usr/lib/x86_64-linux-gnu/pmix/lib"
)

# Link the necessary libraries for the target
target_link_libraries(SU3_QuantumTrajectories PRIVATE
    ${MPI_LIBRARIES}
    OpenMP::OpenMP_CXX
    -lgsl
    -lopenblas
    -llapack
    -lboost_iostreams
    -lboost_system
    -lboost_filesystem
    -lm
    -ldl
    -lpetsc -lslepc
)

