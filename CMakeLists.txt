cmake_minimum_required(VERSION 3.16)
project(SPT_cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(SPT_cpp main.cpp FactorTree.cpp FactorTree.h  SPT.h SPT.cpp
         models.h rv.cpp factor.cpp graph.hpp graph.cpp) 



find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)




set(libDAI_DIR PATH_TO_YOUR_LIBDAI)
message("libDAI_DIR: ${libDAI_DIR}")

include_directories(${libDAI_DIR}/include)

link_directories(${libDAI_DIR}/lib)




include_directories(${LIBDAI_INCLUDE_DIR})

include_directories(${LIBDAI_INCLUDE_DIR})
include_directories(/usr/include/libdai_3_1/libDAI-0.3.1/include /usr/lib/x86_64-linux-gnu)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)
target_link_libraries(SPT_cpp PRIVATE Threads::Threads Eigen3::Eigen OpenMP::OpenMP_CXX  
/usr/lib/x86_64-linux-gnu/libgmpxx.a /usr/lib/x86_64-linux-gnu/libgmp.so  PATH_TO_YOUR/libdai.a)

