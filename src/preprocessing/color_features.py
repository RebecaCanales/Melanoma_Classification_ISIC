import numpy as np

def extract_color_statistics(image):
    mean_colors = np.mean(image, axis=(0,1))
    std_colors = np.std(image, axis=(0,1))
    return {
        "mean_red": mean_colors[2], "mean_green": mean_colors[1], "mean_blue": mean_colors[0],
        "std_red": std_colors[2], "std_green": std_colors[1], "std_blue": std_colors[0]
    }
