cmake_minimum_required(VERSION 3.10)
project(CompiledSolver)

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_library(FATROP_LIB fatrop REQUIRED)
find_library(BLASFEO_LIB blasfeo REQUIRED)

# Add blasfeo include directory (adjust path as necessary)
include_directories(/opt/blasfeo/include)

# Add the source file
set(SOURCE_FILE solver_function.c)

# Create the shared library
add_library(solver_function SHARED ${SOURCE_FILE})

# Link the libraries
target_link_libraries(solver_function PRIVATE ${FATROP_LIB} ${BLASFEO_LIB})

# Compilation flags
target_compile_options(solver_function PRIVATE -O3 -fPIC)

# Install the shared library
install(TARGETS solver_function LIBRARY DESTINATION lib)
