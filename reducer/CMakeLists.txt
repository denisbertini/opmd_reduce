
find_package(openPMD 0.14.5 CONFIG)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wno-narrowing -fopenmp")

SET ( EXECUTABLENAME opmd_reducer )

SET ( ${EXECUTABLENAME}_SRCS
        opmd_reducer.cxx
)

include_directories(
        .
        ${CMAKE_SOURCE_DIR}/include
	${CMAKE_SOURCE_DIR}/merger/	
)

ADD_EXECUTABLE(${EXECUTABLENAME}
        ${${EXECUTABLENAME}_SRCS}
)


TARGET_LINK_LIBRARIES (${EXECUTABLENAME} PRIVATE openPMD::openPMD OPMDREDUCER "stdc++fs")


ADD_TEST(${EXECUTABLENAME} ${EXECUTABLENAME})

INSTALL ( TARGETS ${EXECUTABLENAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


INCLUDE_DIRECTORIES ()
