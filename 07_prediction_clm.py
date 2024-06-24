import os

import pandas as pd
import numpy as np
from osgeo import gdal, ogr, osr
import math
from time import time
from enum import Enum
from joblib import load
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import rasterio.mask
import cv2

from fire_modules.utils import convert_coordinates
from fire_modules import interpolate



class Zona(Enum):
    AB_CU = ['Albacete', 'Cuenca']
    TO_CR_GU = ['Toledo', 'Ciudad Real', 'Guadalajara']


def print_time(string, time):
    formatted_string = string[:30].ljust(30, '.')
    formatted_time = f"{time:7.2f}"
    result = f"{formatted_string} {formatted_time} seg."
    print(result)


climatic_variables = ['anomalia', 'dpv', 'vel_media_viento']


def get_dict_climatic_variables_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    dict_climatic_variables = {}
    for climatic_var in climatic_variables:
        dict_climatic_variables[climatic_var] = df[['lat', 'lon', climatic_var]].dropna().reset_index()
    return dict_climatic_variables



def get_climatic_var_matrix_from_dataframe(x0, y0, x_pixel_size, y_pixel_size, num_rows, num_cols, var, df):
    print(f'\nInterpolando {var}...')

    df[['x','y']] = df.apply(lambda row: pd.Series(convert_coordinates(row['lon'], row['lat'], org='4326', dst='25830')), axis=1)
    x_min_estaciones = df['x'].min()
    y_min_estaciones = df['y'].min()

    x_max_estaciones = df['x'].max()
    y_max_estaciones = df['y'].max()


    x_min_estaciones = min(x_min_estaciones, x0)
    x_max_estaciones = max(x_max_estaciones, x0 + num_cols*x_pixel_size)
    y_min_estaciones = min(y_min_estaciones, y0 + num_rows*y_pixel_size)
    y_max_estaciones = max(y_max_estaciones, y0)

    df_variable = df.copy()
    df_variable.rename(columns={var: 'value'}, inplace=True)

    interpolated_matrix = interpolate.get_interpolated_variable_matrix(x_left_limit=x0, y_upper_limit=y0, x_pixel_size=x_pixel_size, y_pixel_size=y_pixel_size, 
                                                                            num_rows=num_rows, num_cols=num_cols, df_variable=df_variable)
    
    return interpolated_matrix.copy()



def get_X_and_coords_by_shape_matrices(shape_path:str, dict_climatic_variables:dict):
    fire_variables = {} 

    folder_path = os.path.join('data', 'raster_shapes_clm')

    fire_variables['elevacion'] = {}
    fire_variables['elevacion']['path'] = os.path.join(folder_path, 'elevacion_clm.tif')

    fire_variables['orientacion'] = {}
    fire_variables['orientacion']['path'] = os.path.join(folder_path, 'orientacion_clm.tif')

    fire_variables['altura'] = {}
    fire_variables['altura']['path'] = os.path.join(folder_path, 'alturas_clm.tif')

    fire_variables['erodi'] = {}
    fire_variables['erodi']['path'] = os.path.join(folder_path, 'erodi_clm.tif')

    fire_variables['inflam'] = {}
    fire_variables['inflam']['path'] = os.path.join(folder_path, 'inflam_clm.tif')

    fire_variables['mcroth'] = {}
    fire_variables['mcroth']['path'] = os.path.join(folder_path, 'mcroth_clm.tif')

    fire_variables['slope'] = {}
    fire_variables['slope']['path'] = os.path.join(folder_path, 'slope_clm.tif')

    fire_variables['lfcc'] = {}
    fire_variables['lfcc']['path'] = os.path.join(folder_path, 'lfcc_clm.tif')


    shape_ds = ogr.Open(shape_path)
    shape_layer = shape_ds.GetLayer()
    x_limit_left, x_limit_right, y_limit_bottom, y_limit_up = shape_layer.GetExtent()
    x_pixel_size = 25
    y_pixel_size = -25
    num_columns_array_cut = math.ceil(abs((x_limit_right-x_limit_left)/x_pixel_size))
    num_rows_array_cut = math.ceil(abs((y_limit_up-y_limit_bottom)/y_pixel_size))
    
    ############# CALCULAR VARIABLES CLIMATICAS ##############
    fire_variables['anomalia'] = {}
    fire_variables['dpv'] = {}
    fire_variables['vel_media_viento'] = {}

    fire_variables['anomalia']['array'] = get_climatic_var_matrix_from_dataframe(x_limit_left, y_limit_up, x_pixel_size, y_pixel_size, num_rows_array_cut, 
                                                                    num_columns_array_cut, 'anomalia', dict_climatic_variables['anomalia'])
    fire_variables['dpv']['array'] = get_climatic_var_matrix_from_dataframe(x_limit_left, y_limit_up, x_pixel_size, y_pixel_size, num_rows_array_cut, 
                                                              num_columns_array_cut, 'dpv', dict_climatic_variables['dpv'])
    fire_variables['vel_media_viento']['array'] = get_climatic_var_matrix_from_dataframe(x_limit_left, y_limit_up, x_pixel_size, y_pixel_size, num_rows_array_cut, 
                                                                num_columns_array_cut, 'vel_media_viento', dict_climatic_variables['vel_media_viento'])


    ############################## MASCARA PARA PREDECIR CADA PUNTO CON SU MODELO ##############################
    def get_raster_array_from_shapefile(shapefile_path, cols, rows, value):
        # Abrir el shapefile
        shape_ds = ogr.Open(shapefile_path)
        if shape_ds is None:
            print("No se pudo abrir el shapefile.")
            return

        shape_layer = shape_ds.GetLayer()


        # Crear el dataset del raster de salida
        driver = gdal.GetDriverByName('MEM')
        output_raster_ds = driver.Create("", cols, rows, 1, gdal.GDT_Float32)

        # Establecer la información de geotransformación y el sistema de referencia espacial del raster de salida
        output_raster_ds.SetGeoTransform((x_limit_left, x_pixel_size, 0, y_limit_up, 0, y_pixel_size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(25830)
        output_raster_ds.SetProjection(srs.ExportToWkt())

        # Rasterizar el shapefile con el valor dado
        gdal.RasterizeLayer(output_raster_ds, [1], shape_layer, burn_values=[value])

        # Leer el contenido del rasterizado
        raster_array = output_raster_ds.GetRasterBand(1).ReadAsArray()

        # Cerrar los datasets
        shape_ds = None
        output_raster_ds = None

        return raster_array
    ############################################################################################################
    def add_columns_end(np_array:np.array, value, num_columns=1):
        new_columns = np.full((np_array.shape[0], num_columns), value)
        return np.c_[np_array, new_columns]

    def add_rows_end(np_array:np.array, value, num_rows=1):
        new_rows = np.full((num_rows, np_array.shape[1]), value)
        return np.vstack((np_array, new_rows))

    def add_columns_start(np_array:np.array, value, num_columns=1):
        new_columns = np.full((np_array.shape[0], num_columns), value)
        return np.c_[new_columns, np_array]

    def add_rows_start(np_array:np.array, value, num_rows=1):
        new_rows = np.full((num_rows, np_array.shape[1]), value)
        return np.vstack((new_rows, np_array))


    def cut_array_to_shape():
        num_columns_array_cut = math.ceil(abs((x_limit_right-x_limit_left)/x_pixel_size))
        num_rows_array_cut = math.ceil(abs((y_limit_up-y_limit_bottom)/y_pixel_size))
        num_columns_cut_from_original_array = num_columns_array_cut
        num_rows_cut_from_original_array = num_rows_array_cut

        initial_column_cut_from_original = int((x_limit_left - x0) // x_pixel_size)
        initial_row_cut_from_original = int((y_limit_up - y0) // y_pixel_size)


        if initial_column_cut_from_original < 0:
            num_columns_add_start = min(abs(initial_column_cut_from_original), num_columns_array_cut)
            num_columns_cut_from_original_array -= num_columns_add_start
            num_columns_cut_from_original_array = max(0, num_columns_cut_from_original_array)
            initial_column_cut_from_original = 0
        else:
            num_columns_add_start = 0

        if initial_row_cut_from_original < 0:
            num_rows_add_start = min(abs(initial_row_cut_from_original), num_rows_array_cut)
            num_rows_cut_from_original_array -= num_rows_add_start
            num_rows_cut_from_original_array = max(0, num_rows_cut_from_original_array)
            initial_row_cut_from_original = 0
        else:
            num_rows_add_start = 0


        final_column_cut_from_original = initial_column_cut_from_original + num_columns_cut_from_original_array
        final_row_cut_from_original = initial_row_cut_from_original + num_rows_cut_from_original_array

        cut_array = original_array[initial_row_cut_from_original:final_row_cut_from_original, initial_column_cut_from_original:final_column_cut_from_original]

        if num_rows_add_start > 0:
            cut_array = add_rows_start(cut_array, -9999, num_rows_add_start)

        if num_columns_add_start > 0:
            cut_array = add_columns_start(cut_array, -9999, num_columns_add_start)

        rows_cut_array, columns_cut_array = cut_array.shape

        num_columns_add_end = num_columns_array_cut - columns_cut_array
        if num_columns_add_end > 0:
            cut_array = add_columns_end(cut_array, -9999, num_columns_add_end)

        num_rows_add_end = num_rows_array_cut - rows_cut_array
        if num_rows_add_end > 0:
            cut_array = add_rows_end(cut_array, -9999, num_rows_add_end)

        cut_array[cut_array==no_data_value] = np.nan

        return cut_array


    # arrays_variables = {}
    X = pd.DataFrame()


    ################# models mask #################
    time_start_mask = time()
    shapefile_path1 = os.path.join('data', 'shapes', 'TO_CR_GU', 'TO_CR_GU_shape.shp')
    shapefile_path2 = os.path.join('data', 'shapes', 'AB_CU', 'AB_CU_shape.shp')

    mask_models_array = np.full(shape=(num_rows_array_cut, num_columns_array_cut), fill_value=-9999)

    raster_array1 = get_raster_array_from_shapefile(shapefile_path1, num_columns_array_cut, num_rows_array_cut, 1)
    raster_array2 = get_raster_array_from_shapefile(shapefile_path2, num_columns_array_cut, num_rows_array_cut, 2)

    mask_models_array[raster_array1 == 1] = 1
    mask_models_array[raster_array2 == 2] = 2
    
    mask_driver = gdal.GetDriverByName('MEM')
    mask_ds = mask_driver.Create('', mask_models_array.shape[1], mask_models_array.shape[0], 1, gdal.GDT_Int32)
    no_data_value = -9999
    mask_band = mask_ds.GetRasterBand(1)
    mask_band.SetNoDataValue(no_data_value)
    mask_ds.SetGeoTransform((x_limit_left, x_pixel_size, 0, y_limit_up, 0, y_pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(25830)
    mask_ds.SetProjection(srs.ExportToWkt())
    mask_band.WriteArray(mask_models_array)

    mask_ds = gdal.Warp('', mask_ds, cutlineDSName=shape_path, cutlineLayer=str(shape_layer.GetName()), dstNodata=no_data_value, format='MEM')
    mask_band = mask_ds.GetRasterBand(1)


    mask_models_array = mask_band.ReadAsArray()
    mask_models_array = mask_models_array.flatten()
    X['mask_models'] = mask_models_array

    mask_ds = None
    mask_band = None
    shape_ds = None

    total_time_mask = np.round(time()-time_start_mask, 2)
    print_time('Máscara shape y modelos', total_time_mask)
    ###############################################


    for var in fire_variables:
        time_start_variable = time()
        if var not in climatic_variables:
            ds = gdal.Open(fire_variables[var]['path'])
            band = ds.GetRasterBand(1)
            no_data_value = band.GetNoDataValue() 
            
            original_array = band.ReadAsArray()
            transform = ds.GetGeoTransform()
            x0 = transform[0]
            y0 = transform[3]

            ds = None
            band = None
            
            cut_array = cut_array_to_shape()

        else:
            cut_array = fire_variables[var]['array']
            no_data_value = -9999
            cut_array[cut_array==no_data_value] = np.nan

        X[var] = cut_array.flatten()
        
        total_time_variable = np.round(time()-time_start_variable, 2)
        print_time(f'{var} {cut_array.shape}', total_time_variable)

    del(original_array)
    del(cut_array)
    

    return X, num_rows_array_cut, num_columns_array_cut, y_limit_up, x_limit_left




def predict(df:pd.DataFrame, zona:Zona):
    X = df[['elevacion', 'orientacion_cos', 'orientacion_sen', 'altura', 'slope', 'lfcc', 'anomalia', 'dpv', 'vel_media_viento', 'erodi', 'inflam', 'mcroth']]
    X.reset_index(drop=True, inplace=True)
    zona = Zona(zona)
    folder_path = os.path.join('models', 'experiment_4', zona.name)
    
    model = {
        'baja_alta': load(os.path.join(folder_path, 'model_baja_alta.joblib')),
        'baja_mediabaja': load(os.path.join(folder_path, 'model_baja_mediabaja.joblib')),
        'mediaalta_alta': load(os.path.join(folder_path, 'model_mediaalta_alta.joblib')),
    }
    

    pred = model['baja_alta'].predict(X)
    predictions_baja_alta = pred
    predictions_final = np.zeros_like(predictions_baja_alta)

    if np.count_nonzero(predictions_baja_alta == 0) > 0:
        indexes_pred_baja = X.index[predictions_baja_alta == 0]
        pred = model['baja_mediabaja'].predict(X.loc[predictions_baja_alta == 0])

        predictions_baja_media = pred
        predictions_baja_media = np.where(predictions_baja_media == 0, 0, 1)
        predictions_final[indexes_pred_baja] = predictions_baja_media  

    if np.count_nonzero(predictions_baja_alta == 1) > 0:
        indexes_pred_alta = X.index[predictions_baja_alta == 1]
        pred = model['mediaalta_alta'].predict(X.loc[predictions_baja_alta == 1])

        predictions_alta_muyalta = pred
        predictions_alta_muyalta = np.where(predictions_alta_muyalta == 0, 2, 3)
        predictions_final[indexes_pred_alta] = predictions_alta_muyalta

    return predictions_final



def generate_predictions_raster_matricial(predictions, x0, y0, output_raster_path, x_pixel_size=25, y_pixel_size=-25):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_raster_path, predictions.shape[1], predictions.shape[0], 1, gdal.GDT_Float32)

    no_data_value = -9999
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)

    dataset.SetGeoTransform((x0, x_pixel_size, 0, y0, 0, y_pixel_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(25830)
    dataset.SetProjection(srs.ExportToWkt())

    band.WriteArray(predictions)

    dataset = None




def raster_to_png(input_raster_path, output_png_path, shapefile_path, max_size=2048):
    def resize_image(image_path, output_path, max_size):
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        scale = min(max_size/width, max_size/height)
        resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)

    with rasterio.open(input_raster_path) as src:
        raster_data, _ = rasterio.mask.mask(src, gpd.read_file(shapefile_path).geometry, crop=True, nodata=5)
       
        raster_data = np.nan_to_num(raster_data, nan=4)
        
        raster_data = (raster_data / np.nanmax(raster_data)) * 255
        raster_data = raster_data.astype(np.uint8)

        cmap = ListedColormap([
            (171/255,221/255,164/255), #0
            (255/255,255/255,191/255), #1
            (253/255,174/255,97/255), #2
            (215/255,25/255,28/255),  #3
            (225/255,225/255,225/255), #4
            (1, 1, 1)
        ])
        
        plt.imsave(output_png_path, raster_data[0], cmap=cmap)
        resize_image(output_png_path, output_png_path, max_size)









if __name__ == '__main__':
    shape_path = os.path.join('data', 'shapes', 'CLM', 'CLM_shape.shp')
    csv_climatic_variables_path = os.path.join('data', 'climatic_variables_clm_15-07-2023.csv')
        

    initial_time = time()
    partial_time = initial_time

    dict_climatic_variables = get_dict_climatic_variables_from_csv(csv_climatic_variables_path)

    X, num_rows, num_cols, y0, x0 = get_X_and_coords_by_shape_matrices(shape_path, dict_climatic_variables)

    orientation_time = time()
    X['orientacion_rad'] = np.deg2rad(X['orientacion'])
    X['orientacion_sen'] = np.sin(X['orientacion_rad'])
    X['orientacion_cos'] = np.cos(X['orientacion_rad'])
    print_time('Calcular orientación', time()-orientation_time)

    clean_time = time()
    X.dropna(inplace=True)
    X = X[
            (X['inflam'] != 0) & 
            (X['mcroth'] != 0) & 
            (X['altura'] != 0) & 
            (X['lfcc'] != 0)
        ]
    print_time('Quitar 0 y NaN', time()-clean_time)


    indexes_time = time()
    indexes_to_predict = X.index
    mask_models_series = X['mask_models']
    X = X.drop('mask_models', axis=1)

    indexes_model_TO_CR_GU = mask_models_series[mask_models_series==1].index
    indexes_model_AB_CU = mask_models_series[mask_models_series==2].index
    print_time('Calcular indices', time()-indexes_time)


    print_time('Dataframes obtenidos', time()-partial_time)


    partial_time = time()
    predictions_array = np.full(num_rows*num_cols, np.nan)
    if len(indexes_model_TO_CR_GU) > 0:
        predictions = predict(X.loc[indexes_model_TO_CR_GU], Zona.TO_CR_GU)
        predictions_array[indexes_model_TO_CR_GU] = predictions
    if len(indexes_model_AB_CU) > 0:
        predictions = predict(X.loc[indexes_model_AB_CU], Zona.AB_CU)
        predictions_array[indexes_model_AB_CU] = predictions

    predictions_array = predictions_array.reshape(num_rows, num_cols)

    print_time('Predicciones', time()-partial_time)
    partial_time = time()


    output_raster_path = os.path.join('pred_clm', 'pred_clm_15-07-2023.tif')
    generate_predictions_raster_matricial(predictions_array, x0, y0, output_raster_path)

    print_time('Raster creado', time()-partial_time)
    partial_time = time()


    output_png_path = os.path.join('pred_clm', 'pred_clm_15-07-2023.png')
    raster_to_png(output_raster_path, output_png_path, shape_path)
    print_time('Raster a imagen', time()-partial_time)
    partial_time = time()


    total_time = time()-initial_time
    print_time('Tiempo total', total_time)

