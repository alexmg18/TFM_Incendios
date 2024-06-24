#!/bin/bash

# Establece el número de threads en 1 si no se proporciona un segundo parámetro
NUM_THREADS=${1:-1}

# Ejecuta el programa con el número de threads especificado y las filas como argumento
OMP_NUM_THREADS=$NUM_THREADS /home/openmp/interpolate_omp
