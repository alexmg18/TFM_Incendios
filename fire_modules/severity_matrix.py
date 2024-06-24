import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from fire_modules.utils import Colors


def get_severity_matrix(severities:np.array, coords:pd.DataFrame):
    coords_x = coords['coord_x_etrs89']
    coords_y = coords['coord_y_etrs89']

    x_uniques = np.sort(coords_x.unique())
    y_uniques = np.sort(coords_y.unique())

    x_pixel_size = min(x_uniques[1:] - x_uniques[:-1])
    y_pixel_size = min(y_uniques[1:] - y_uniques[:-1])

    x_min = coords_x.min() -0.01
    x_max = coords_x.max() +0.01
    y_min = coords_y.min() -0.01
    y_max = coords_y.max() +0.01

    num_rows = int((y_max - y_min) // y_pixel_size) + 1
    num_cols = int((x_max - x_min) // y_pixel_size) + 1

    matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)

    for value, coord_x, coord_y in zip(severities, coords_x, coords_y):
            row = int((y_max - coord_y) // y_pixel_size)
            col = int((coord_x - x_min) // x_pixel_size)

            matrix[row, col] = value

    return matrix



def draw_axes_with_severity_matrix(axes:plt.axes, matrix:np.array, colors:Colors, title, nan_color='#e8e8e8'):
    colors_dict = colors.value
    cmap = ListedColormap([colors_dict[x] for x in sorted(colors_dict.keys())])

    labels = np.array(sorted(colors_dict.keys()))
    len_lab = len(labels)

    norm_bins = np.sort([*colors_dict.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
    fmt = plt.FuncFormatter(lambda x, pos: labels[norm(x)])

    axes.set_facecolor(nan_color)
    im = axes.imshow(matrix, cmap=cmap, norm=norm)

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    plt.colorbar(im, format=fmt, ticks=tickz, location='bottom', pad=0.05)

    axes.set_title(title)

    axes.set_xticks([])
    axes.set_yticks([])


def show_original_prediction_evaluation_severity_matrices(original_matrix:np.array, predicted_matrix:np.array, figsize=(20,12), fig_title=''):
    evaluation_matrix = predicted_matrix - original_matrix

    axes_grid = np.zeros(3, dtype=object)
    fig = plt.figure(figsize=figsize)
    
    
    axes_grid[0] = plt.subplot2grid((2,2), (0,0))
    axes_grid[1] = plt.subplot2grid((2,2), (0,1))
    axes_grid[2] = plt.subplot2grid((2,4), (1,1), colspan=2)

    for axes in axes_grid:
        fig.add_axes(axes)

    draw_axes_with_severity_matrix(axes_grid[0], original_matrix, Colors.Severity, 'Original')
    draw_axes_with_severity_matrix(axes_grid[1], predicted_matrix, Colors.Severity, 'Predicción')
    draw_axes_with_severity_matrix(axes_grid[2], evaluation_matrix, Colors.Evaluation, 'Evaluación')

    fig.subplots_adjust(wspace=0.2, hspace=0.1)  

    fig.suptitle(fig_title)

    # evaluation_text = f'Evaluación\n\n'
    # unique_values, frequencies = np.unique(evaluation_matrix[~np.isnan(evaluation_matrix)], return_counts=True)
    # total = frequencies.sum()
    # for value, frequency in zip(unique_values, frequencies):
    #     evaluation_text += f'{int(value):>2}: {round((frequency/total)*100, 1)}% - ({frequency:,} / {total:,})\n'

    # x_pos = 1.25
    # y_pos = 1
    # axes_grid[2].text(x_pos, y_pos, evaluation_text, fontsize=12, ha='left', va='top', transform=axes_grid[2].transAxes)


