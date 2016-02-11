mkdir -p ../SIMDNumberCMake-Release
cd ../SIMDNumberCMake-Release
cmake -DCMAKE_BUILD_TYPE=Release ../SIMDNumber
make -j 2
ln -s `realpath ./SIMDTest` ../SIMDNumber/SIMDTest-release

mkdir -p ../SIMDNumberCMake-Debug
cd ../SIMDNumberCMake-Debug
cmake -DCMAKE_BUILD_TYPE=Debug ../SIMDNumber
make -j 2
ln -s `realpath ./SIMDTest` ../SIMDNumber/SIMDTest-debug
