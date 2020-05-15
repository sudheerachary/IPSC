- Problem-1a
    - g++ -fopenmp problem-1a.cpp -o problem-1a
    - OMP_NUM_THREADS=10 ./problem-1a

- Problem-1b
    - mpic++ -o3 problem-1b.cpp -o problem-1b
    - mpirun -np 3 problem-1b

- Problem-1c
    - nvcc problem-1c.cu -o problem-1c
    - ./problem-1c

- Problem-2
    - g++ -fopenmp problem-2.cpp -o problem-2
    - OMP_NUM_THREADS=4 ./problem-2

- Problem-3a
    - g++ -fopenmp problem-3a.cpp -o problem-3a
    - OMP_NUM_THREADS=10 ./problem-3a

- Problem-3b
    - mpic++ -o3 problem-3b.cpp -o problem-3b
    - mpirun -np 6 problem-3b

- Problem-3c
    - nvcc problem-3c.cu -o problem-3c
    - ./problem-3c

- Problem-4
    - g++ -fopenmp problem-4.cpp -o problem-4
    - OMP_NUM_THREADS=10 ./problem-4
    
- Problem-5
    - nvcc problem-5.cu -o problem-5
    - ./problem-5

- Problem-6
    - nvcc `pkg-config --cflags opencv` problem-6.cu `pkg-config --libs opencv` -o problem-6
    - ./problem-6