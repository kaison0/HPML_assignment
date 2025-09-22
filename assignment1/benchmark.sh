#!/bin/bash

echo "Comiling"
gcc -O3 -Wall dp1 dp1.c
gcc -O3 -Wall dp2 dp2.c
gcc dp3.c -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 \
    -lmkl_rt -lpthread -lm -ldl -O3 -Wall dp3
echo "Runing benchmarks..."
echo "----C1 1000000 1000"
./dp1 1000000 1000
echo "----C1 300000000 20"
./dp1 300000000 20
echo "----C2 1000000 1000"
./dp2 1000000 1000
echo "----C2 300000000 20"
./dp2 300000000 20
echo "----C3 1000000 1000"
./dp3 1000000 1000
echo "----C3 300000000 20"
./dp3 300000000 20
echo "----C4 1000000 1000"
python3 dp4.py 1000000 1000
echo "----C4 300000000 20"
python3 dp4.py 300000000 20
echo "----C5 1000000 1000"
python3 dp5.py 1000000 1000
echo "----C5 300000000 20"
python3 dp5.py 300000000 20

echo "All done!"
