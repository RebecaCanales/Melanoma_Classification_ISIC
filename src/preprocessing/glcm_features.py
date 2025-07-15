from skimage.feature import greycomatrix, greycoprops

def extract_glcm_features(gray_image):
    distances = [1, 2, 3]
    angles = [0, 3.14/4, 3.14/2, 3*3.14/4]
    glcm = greycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast').mean()
    homogeneity = greycoprops(glcm, 'homogeneity').mean()
    return {"GLCM_contrast": contrast, "GLCM_homogeneity": homogeneity}
