#!/bin/bash

OMP_NUN_THREADS=28 srun -N 1 --cpus-per-task=28 ./omp_sched

