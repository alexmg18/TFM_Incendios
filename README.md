# Predicción de la severidad de incendios mediante técnicas de aprendizaje automático

Trabajo Fin de Máster

Máster Universitario en Ingeniería Informática.

Autor: __Alejandro Martínez García__

# Instrucciones para ejecutar el proyecto

## Requisitos previos

### Archivos grandes
Los archivos que pesan más de 100 MB están en otro repositorio en OneDrive en este [enlace](https://pruebasaluuclm-my.sharepoint.com/:f:/g/personal/alejandro_martinez39_alu_uclm_es/EuTE_auwQ8RDrZEGrQTNLW8B7IDmwRbfsfO4IYwqfv0knw). Estos archivos son:

- La carpeta `raster_shapes_clm` que contiene las capas ráster con los datos de Castilla-La Mancha. Esta carpeta debe incluirse en la ruta del proyecto `data/`.
- La carpeta `experiment_1` que contiene los dos modelos entrenados en el primer experimento. Esta carpeta debe reemplazar a la ya existente en la ruta del proyecto `models/`.
- El archivo `pred_clm/pred_clm_15-07-2023.tif`es el ráster con la predicción de Castilla-La Mancha generada en la sección. Este archivo debe incluirse en la ruta del proyecto `pred_clm/`.

### Entorno virtual de Python
Para ejecutar el código Python se ha hecho uso de un entorno virtual de Anaconda, donde se define la versión 3.12 de Python y las bibliotecas usadas. Este entorno se encuentra exportado en el archivo `tfm_env.yml`.

Para crear el entorno virtual, es necesario tener instalado Anaconda o Miniconda, cuyo procedimiento según el sistema operativo que se use está detallado en la página web oficial. Una vez instalado, simplemente hay que acceder a la carpeta del proyecto y ejecutar el siguiente comando:

```bash
conda env create -f tfm_env.yml
```

### Docker
Para ejecutar el algoritmo de interpolación con OpenMP, se ha hecho uso de un contenedor Docker para evitar problemas de incompatibilidad entre sistemas operativos. El Dockerfile con el que se va a construir la imagen de Docker se ha obtenido del repositorio de GitHub del siguiente [enlace](https://github.com/spagnuolocarmine/docker-mpi).

Para crear el contenedor es necesario tener instalado Docker, cuyo procedimiento según el sistema operativo que se use está detallado en la página web oficial. Una vez instalado, hay que acceder a la carpeta `docker_mpi_openmp/` del proyecto y seguir los siguiente pasos:

- Construir la imagen, que se va a llamar `img_docker_mpi_openmp`.
```bash
docker build --no-cache -t img_docker_mpi_openmp .
```
- Ejecutar el contenedor a partir de la imagen, con el nombre `mpi_openmp`, y montar un volumen de la carpeta actual sobre la carpeta `/home` dentro del contenedor.
```bash
docker run -d --name mpi_openmp -v .:/home -t img_docker_mpi_openmp
```
- Compilar el programa.
```bash
docker exec mpi_openmp /bin/bash -c "cd /home/openmp && make"
```


## Orden de ejecución
El orden de ejecución de las libretas y programas es:

1. `01_generate_csv_severities.ipynb`. En esta libreta se hace la recopilación de los datos.
2. `02_data_analysis.ipynb`. En esta libreta se hace la transformación y el análisis de los datos.
3. `03_experiment_1.ipynb`. En esta libreta se desarrolla el primer experimento.
4. `04_experiment_2.ipynb`. En esta libreta se desarrolla el segundo experimento.
5. `05_experiment_3.ipynb`. En esta libreta se desarrolla el tercer experimento.
6. `06_experiment_4.ipynb`. En esta libreta se desarrolla el cuarto experimento.
7. `07_prediction_clm.py`. En este programa se hace la predicción de Castilla-La Mancha para un día concreto.
