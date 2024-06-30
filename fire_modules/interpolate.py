import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_PATH)

import numpy as np
import pandas as pd
import subprocess

from fire_modules.utils import load_matrix_bin, save_matrix_bin, convert_coordinates


def interpolate_openmp(matrix:np.array, known_rows:np.array, known_cols:np.array, known_values:np.array, n_processes=8):
    def save_rows_cols_values_bin(path:str, known_rows:np.array, known_cols:np.array, known_values:np.array):
        num_values = known_rows.size
        with open(path, 'wb') as f:
            f.write(num_values.to_bytes(4, byteorder='little')) # 4 bytes for integers of 32 bits
            f.write(known_rows.tobytes())
            f.write(known_cols.tobytes())
            f.write(known_values.tobytes())

    path_c = os.path.expanduser(os.path.join('docker_mpi_openmp', 'openmp', 'data'))
    input_matrix_path = os.path.join(path_c, 'input_matrix_c.dat')
    output_matrix_path = os.path.join(path_c, 'output_matrix_c.dat')
    input_known_values_path = os.path.join(path_c, 'known_values.dat')

    save_rows_cols_values_bin(input_known_values_path, known_rows, known_cols, known_values)
    save_matrix_bin(input_matrix_path, matrix)

    print('**********************************************************************')
    command = ['docker', 'exec', 'mpi_openmp', '/bin/bash', '-c', 'OMP_NUM_THREADS=8 /home/openmp/interpolate_omp']
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.stdout: print(result.stdout)
    if result.stderr: print('Error:\n', result.stderr)
    print('**********************************************************************')

    matrix_interpolated = load_matrix_bin(output_matrix_path)

    return matrix_interpolated


def get_interpolated_variable_matrix(x_left_limit, y_upper_limit, x_pixel_size, y_pixel_size, num_rows, num_cols, df_variable):
    x_right_limit = x_left_limit + x_pixel_size * num_cols
    y_lower_limit = y_upper_limit + y_pixel_size * num_rows
    def get_total_limit(shape_limit, stations_limit, function, pixel_size):
        if function == min:
            total_limit = min(shape_limit, stations_limit)
            if total_limit < shape_limit:
                total_limit -= (total_limit - shape_limit) % abs(pixel_size)
        elif function == max:
            total_limit =  max(shape_limit, stations_limit)
            if total_limit > shape_limit:
                total_limit += (abs(pixel_size) - (total_limit - shape_limit)) % abs(pixel_size)
        
        return total_limit

    df_variable[['coord_x_etrs89','coord_y_etrs89']] = df_variable.apply(lambda row: pd.Series(convert_coordinates(row['lon'], row['lat'], 4326, 25830)), axis=1)

    x_left_limit_station = df_variable['coord_x_etrs89'].min()
    x_right_limit_station = df_variable['coord_x_etrs89'].max()
    y_upper_limit_station = df_variable['coord_y_etrs89'].max()
    y_lower_limit_station = df_variable['coord_y_etrs89'].min()


    x_left_limit_total = get_total_limit(x_left_limit, x_left_limit_station, min, x_pixel_size)
    x_right_limit_total = get_total_limit(x_right_limit, x_right_limit_station, max, x_pixel_size)
    y_upper_limit_total = get_total_limit(y_upper_limit, y_upper_limit_station, max, y_pixel_size)
    y_lower_limit_total = get_total_limit(y_lower_limit, y_lower_limit_station, min, y_pixel_size)


    num_rows_total = int(abs((y_upper_limit_total - y_lower_limit_total) // y_pixel_size) + 1)
    num_cols_total = int(abs((x_right_limit_total - x_left_limit_total) // x_pixel_size) + 1)

    matrix_total = np.full((num_rows_total, num_cols_total), np.nan)

    known_rows, known_cols, known_values = [], [], []
    for _, station_row in df_variable.iterrows():
        row = int((station_row['coord_y_etrs89'] - y_upper_limit_total) // y_pixel_size)
        col = int((station_row['coord_x_etrs89'] - x_left_limit_total) // x_pixel_size)
        matrix_total[row, col] = station_row['value']
        known_rows.append(row)
        known_cols.append(col)
        known_values.append(station_row['value'])

    # Calculate the matrix of the original shape to interpolate
    initial_row = int((y_upper_limit - y_upper_limit_total) // y_pixel_size)
    initial_col = int((x_left_limit - x_left_limit_total) // x_pixel_size)
    matrix = matrix_total[initial_row:initial_row+num_rows, initial_col:initial_col+num_cols]

    known_rows = (np.array(known_rows) - initial_row).astype(np.int32)
    known_cols = (np.array(known_cols) - initial_col).astype(np.int32)
    known_values = np.array(known_values, dtype=np.float64)
    
    matrix_interpolated = interpolate_openmp(matrix, known_rows=known_rows, known_cols=known_cols, known_values=known_values, n_processes=8)
    return matrix_interpolated