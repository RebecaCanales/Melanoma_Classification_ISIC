# Melanoma_Classification_ISIC
Classification of skin lesions as benign and malignant (melanoma).

Objective

The main objectives of this project were to investigate if the feature presented in the feature extraction section contribute to improve accuracy on classification of benign and 
malignant (melanoma) skin tumors using different traditional machine learning methods, as well as compare their performance.

Dataset

The dataset was obtained from the International Skin Imaging Collaboration (ISIC) database, it consists of 44108 dermoscopic images in RGB color space from 2746 patients, 584
correspond to malignant skin tumor – melanoma – and 32542 to benign tumors – letingo NOS, lichenoid keratosis, nevus, seborrheic keratosis, solar letingo –. The images were 
obtained from torso, lower and upper extremity, head, neck, palms, soles, oral and genital zones. For this project, only a subset of this dataset was used, due to computation 
limitations, corresponding to 4391 images – 509 malignant and 3882 benign –, from patients with 10-90 years, 2190 males and 2201 females.

Methodology

a) Image processing. - Images were resized to 1024x1024 pixels. After that, I proceeded to remove the hair with the following process: convert to grayscale, apply black-hat 
filter, and finally inpaint original and black-hat images. Otsu threshold coupled with morphological operations were used to segment two regions of interest (ROI) and the 
background, the first ROI was the lesion and the second was the surrounding of the region.

b) Feature extraction. - Color and texture – gray-level co-occurrence matrix (GLCM) – features were extracted. Regarding GLCM features, extraction was performed at three different 
resolutions. A total of 254 features were obtained.

Feature                    Description                                   No. of Features
Color     Color diversity per RGB, HSV and LAB channel                         9
          Mean per RGB, HSV and LAB channel – lesion                           9
          Mean per RGB, HSV and LAB channel – contour                          9
          Std per RGB, HSV and LAB channel – lesion                            9
          Std per RGB, HSV and LAB channel – contour                           9
          Kurtosis per RGB, HSV and LAB channel – lesion                       9
          Kurtosis per RGB, HSV and LAB channel – contour                      9
          Skewness per RGB, HSV and LAB channel – lesion                       9
          Skewness per RGB, HSV and LAB channel – contour                      9
          Centroid distance per RGB, HSV and LAB channel                       9
          LUV histograms L1-norm                                               1
          LUV histograms L2-norm                                               1
Texture   GLCM contrast per RGB, HSV and LAB channel                          27
          GLCM correlation per RGB, HSV and LAB channel                       27
          GLCM homogeneity per RGB, HSV and LAB channel                       27
          GLCM dissimilarity per RGB, HSV and LAB channel                     27
          GLCM angular second moment (ASM) per RGB, HSV and LAB channel       27
          GLCM energy per RGB, HSV and LAB channel                            27

c) Data transformation. - The Yeo-Johnson transformation [48] was applied to obtain a normal distribution for each feature, they were also normalized. Data extracted from lesion 
was adjusted with the background.

d) Classification. - The data was split in 70% training and 30% testing sets. The features were selected using Wilcoxon test. The classification was performed using five 
methods: SVM, KNN, naive bayes, adaboost and random forest and their performance was compared. Cross validation with 200 repetitions was used.
