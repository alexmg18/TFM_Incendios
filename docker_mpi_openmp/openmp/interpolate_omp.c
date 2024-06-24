#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>

void print_matrix(int rows, int cols, double *matrix) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_array(int size, double *array) {
    for (int i = 0; i < size; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}


void save_matrix_txt(double *matrix, int rows, int cols, char *filename){
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        printf("No se pudo abrir el archivo para guardar la matriz.\n");
    }
    else {
        int index = 0;
        for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
                fprintf(file, "%lf ", matrix[index]);
                index++;
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

void save_matrix_bin(double *matrix, int rows, int cols, char *filename){
    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        printf("No se pudo abrir el archivo para guardar la matriz.\n");
    }
    else{
        fwrite(&rows, sizeof(int), 1, file);
        fwrite(&cols, sizeof(int), 1, file);
        fwrite(matrix, sizeof(double), rows * cols, file);
        fclose(file);
    }
}


void load_matrix_txt(double *matrix, int rows, int cols, char *filename){
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("No se pudo abrir el archivo para cargar la matriz.\n");
    }
    else {
        double value;
        int index = 0;
        int max_index = rows * cols;

        while (fscanf(file, "%lf", &value) == 1) {
        matrix[index] = value;
        index ++;

        if (index == max_index) {
            break;
        }
    }

    fclose(file);
    }
}

void get_matrix_dimension(int *rows, int *cols, char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("No se pudo abrir el archivo para cargar la matriz.\n");
    }
    else{
        // fread(rows, sizeof(int), 1, file);
        // fread(cols, sizeof(int), 1, file);
        if (fread(rows, sizeof(int), 1, file) != 1) {}
        if (fread(cols, sizeof(int), 1, file) != 1) {}
        fclose(file);
    }
}

void load_matrix_bin(double *matrix, int rows, int cols, char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("No se pudo abrir el archivo para cargar la matriz.\n");
    }
    else{
        int aux;
        if (fread(&aux, sizeof(int), 1, file) != 1) {}
        if (fread(&aux, sizeof(int), 1, file) != 1) {}
        if (fread(matrix, sizeof(double), rows * cols, file) != 1) {}
        fclose(file);
    }
}


void get_num_cells_with_values(int *num_cells_with_values, char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("No se pudo abrir el archivo para cargar los valores conocidos.\n");
    }
    else{
        if (fread(num_cells_with_values, sizeof(int32_t), 1, file) != 1) {}
        fclose(file);
    }
}
void load_rows_cols_values_bin(double *values, int *rows_with_values, int *cols_with_values, int num_cells_with_values, char *filename){
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("No se pudo abrir el archivo para cargar la valores conocidos: %s\n", filename);
    }
    else{
        int aux;
        if (fread(&aux, sizeof(int), 1, file) != 1) {}
        if (fread(rows_with_values, sizeof(int32_t), num_cells_with_values, file) != 1) {}
        if (fread(cols_with_values, sizeof(int32_t), num_cells_with_values, file) != 1) {}
        if (fread(values, sizeof(double), num_cells_with_values, file) != 1) {}
        fclose(file);
    }
}

void interpolation(double *matrix, int n_cells, int cols, double *values, int *rows_with_values, int *cols_with_values, int num_cells_with_values) {
    #pragma omp parallel for //schedule(dynamic, 100)
    for (int i = 0; i < n_cells; i++) {
        // if (matrix[i] == 0) {
        if (isnan(matrix[i])) {
            int row_index = i / cols;
            int col_index = i % cols;
            double zi = 0.0;
            double total_weight = 0.0;
            for (int j = 0; j < num_cells_with_values; j++) {
                double distance = sqrt(pow((rows_with_values[j] - row_index), 2) + pow((cols_with_values[j] - col_index), 2));
                double weight = 1.0 / pow(distance, 2);
                zi += values[j] * weight;
                total_weight += weight;
            }
            matrix[i] = zi / total_weight;
        }
    }
}

int main(int argc, char *argv[]) {
    
    double start, end, partial_start, partial_end, total_time;

    // Empieza a contar el tiempo
    start = omp_get_wtime();
    partial_start = omp_get_wtime();

    int rows, cols, len_matrix;
    char *input_filename = "/home/openmp/data/input_matrix_c.dat";
    get_matrix_dimension(&rows, &cols, input_filename);
    len_matrix = rows * cols;
    
    int num_cells_with_values;

    
    // Memory allocation
    double *matrix = malloc(rows * cols * sizeof(double));

    input_filename = "/home/openmp/data/input_matrix_c.dat";
    load_matrix_bin(matrix, rows, cols, input_filename);

    input_filename = "/home/openmp/data/known_values.dat";
    get_num_cells_with_values(&num_cells_with_values, input_filename);

    int nthreads;
    #pragma omp parallel
    {
        nthreads=omp_get_num_threads();
    }

    printf("OPENMP\n");
    printf("NUM_PROCESSES: %d\n", nthreads);
    printf("Matrix size: %d x %d \n", rows, cols);
    printf("Num known values: %d\n", num_cells_with_values);
    printf("\n");

    partial_end = omp_get_wtime();
    total_time = partial_end - partial_start;
    printf("Time: %lf seg. (loading matrix)\n", total_time);
    partial_start = omp_get_wtime();

    


    double *values = malloc(num_cells_with_values * sizeof(double));
    int *rows_with_values = malloc(num_cells_with_values * sizeof(int));
    int *cols_with_values = malloc(num_cells_with_values * sizeof(int));


    input_filename = "/home/openmp/data/known_values.dat";
    load_rows_cols_values_bin(values, rows_with_values, cols_with_values, num_cells_with_values, input_filename);


    // Matrix interpolation
    interpolation(matrix, len_matrix, cols, values, rows_with_values, cols_with_values, num_cells_with_values);


    partial_end = omp_get_wtime();
    total_time = partial_end - partial_start;
    printf("Time: %lf seg. (calculating interpolated matrix)\n", total_time);
    partial_start = omp_get_wtime();

    // printf("Matrix_Result:\n");
    // print_matrix(rows, cols, matrix);
    // printf("\n-------------------------------------\n\n");

    char *filename = "/home/openmp/data/output_matrix_c.dat";
    save_matrix_bin(matrix, rows, cols, filename);

    partial_end = omp_get_wtime();
    total_time = partial_end - partial_start;
    printf("Time: %lf seg. (saving matrix)\n", total_time);

    // Free resources
    free(matrix);

    // Termina de contar el tiempo
    end = omp_get_wtime();
    total_time = end - start;
    printf("--------------------------\n");
    printf("Total Time: %lf seg.\n\n", total_time);

    return 0;
}
