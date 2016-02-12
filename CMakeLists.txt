project (SIMDNumbers)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -mavx")

add_executable(SIMDTest
    SIMDTest.cpp
)
