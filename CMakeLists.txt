cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(submodules/pybind11)

# python package ----------------------------------------------
project(Mitc3FEM)

pybind11_add_module(Mitc3FEM src/pybind.cpp)
target_include_directories(Mitc3FEM PUBLIC
        submodules/eigen
        submodules/pybind11/include
        # TODO : include Python.h in this directory
        /usr/include/python3.10)

# c++ test ----------------------------------------------------
project(Mitc3FEMTestCpp)

add_subdirectory(submodules/googletest)
add_executable(Mitc3FEMTestCpp test/test_cpp.cpp)
target_link_libraries(Mitc3FEMTestCpp gtest_main)
target_include_directories(Mitc3FEMTestCpp PUBLIC submodules/eigen)
