cmake_minimum_required(VERSION 3.2...3.27 FATAL_ERROR)
project(OPMDREDUCER C CXX)

# Compilers
enable_language(CXX)

# MPI is mandatory
find_package(MPI REQUIRED)
add_definitions( -DOMPI_SKIP_MPICXX -fpermissive)

# OpenMP
find_package(OpenMP REQUIRED)
# Compilation options
set(CMAKE_CXX_FLAGS "-fPIC -Wshadow -Wall -W -Woverloaded-virtual -fsigned-char -Wextra -std=c++17 -g -fopenmp")

message(STATUS "MPI_INCLUDE_PATH : ${MPI_INCLUDE_PATH}")
message(STATUS "C++ compiler is : ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler Flags are : ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ compiler identification is : ${CMAKE_CXX_COMPILER_ID}")


include_directories(include
        ${MPI_INCLUDE_PATH}  
	include/
	}
)

link_directories(${OTHER_LINK_DIR})

set( OPMDREDUCER_HEADERS
	include/cxxopts.hpp
	include/utils.h
)

set( OPMDREDUCER_SRCS
	merger/vranic_merger.cxx
)

add_library(OPMDREDUCER SHARED ${OPMDREDUCER_SRCS})

target_link_libraries(OPMDREDUCER
   PRIVATE
   MPI::MPI_C
   ${MPI_LIBRARIES}
   )

set_target_properties(OPMDREDUCER PROPERTIES LINKER_LANGUAGE CXX)


# Add executables
add_subdirectory(reducer)

install(
        TARGETS OPMDREDUCER
        LIBRARY DESTINATION lib
)


