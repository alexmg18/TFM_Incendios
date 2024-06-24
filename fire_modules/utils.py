import pyproj
import numpy as np
from enum import Enum


def convert_coordinates(coord_x_longitude, coord_y_latitud, org, dst):
    crs_org = pyproj.CRS.from_epsg(org)
    crs_dst = pyproj.CRS.from_epsg(dst)

    transformer = pyproj.Transformer.from_crs(crs_org, crs_dst, always_xy=True)
    new_x_longitude, new_y_latitude = transformer.transform(coord_x_longitude, coord_y_latitud)

    return new_x_longitude, new_y_latitude


def save_matrix_bin(path:str, matrix:np.array):
    with open(path, 'wb') as f:
        for dim in matrix.shape:
            f.write(dim.to_bytes(4, byteorder='little')) # 4 bytes for integers of 32 bits
        f.write(matrix.tobytes())
    
def load_matrix_bin(path:str):
    with open(path, 'rb') as f:
        rows = int.from_bytes(f.read(4), byteorder='little')
        cols = int.from_bytes(f.read(4), byteorder='little')
        matrix = np.frombuffer(f.read(), dtype=np.float64)
    matrix = matrix.reshape(rows, cols)
    return matrix


class Colors(Enum):
    Severity = {
        0: '#abdda4',
        1: '#ffffbf',
        2: '#fdae61',
        3: '#d7191c'
    }

    Evaluation = {
        -3: '#005187',
        -2: '#84b6f4',
        -1: '#c4dafa',
        0: '#4cb24c',
        1: '#ef9a9a',
        2: '#ef5350',
        3: '#d10d09'
    }