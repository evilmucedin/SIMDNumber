#include <iostream>

#include "SIMDNumber.h"

using namespace std;

int main() {
    SIMDNumber x(0, 1, 2, 3, 4, 5, 6, 7);
    std::cout << x << std::endl;

    return 0;
}
