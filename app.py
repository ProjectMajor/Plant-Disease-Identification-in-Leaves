from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import streamlit as st
import numpy as np
from sklearn import preprocessing
import joblib
st.set_page_config(layout="wide")
col1,col2,col3 = st.columns([1,3,1])
with col2:
	st.title("Plant Disease Identification in Leaves")
st.divider()
col1,col2,col3 = st.columns([1,0.5,1])
with col2:
	logo = Image.open("Logo.jpg")
	st.image(logo)
col1,col2,col3 = st.columns([2,4,1])
with col2:
	st.markdown("##### Plant Leaf Diseases: Undermining Crop Health and Yield")
col1,col2,col3 = st.columns([1,8,1])
with col2:
	st.markdown("(Let's Raise Awareness, Foster Research, and Implement Care for Plant Diseases. Together, We Can Cultivate Healthier Crops and Agricultural Sustainability)")
st.divider()
st.header("Different Plant Diseases")
col3,col4,col5 = st.columns([1,4,1])
with col4:
	stages = Image.open("Diseases.png")
	st.image(stages)
st.divider()
st.header("Upload Leaf for Diagnosis")
img = st.file_uploader("")
op = {0:"Result: Alternaria Alternata",1:"Result: Anthracnose",2:"Result: Bacterial Blight",3:"Result: Healthy"}

# Load and preprocess images
def preprocess_image(image, target_size=(224, 224)):
    gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    resized_image = cv.resize(gray_image, target_size)
    return resized_image
def get_features(image):
    pil_image = Image.open(image).convert('RGB')
    open_cv_image = np.array(pil_image)
    gray = preprocess_image(open_cv_image)
        
    # distance 1 and angle 0
    glcm = graycomatrix(np.array(gray), [1], angles=[0], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_1_0 = (contrast[0][0])
    energy_1_0=(energy[0][0])
    homogeneity_1_0=(homogeneity[0][0])
    correlation_1_0=(correlation[0][0])
        
    # distance 1 and angle 90
        
    glcm = graycomatrix(np.array(gray), [1], angles=[np.pi/2], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_1_90=(contrast[0][0])
    energy_1_90=(energy[0][0])
    homogeneity_1_90=(homogeneity[0][0])
    correlation_1_90=(correlation[0][0])
        
    # distance 3 and angle 0
        
    glcm = graycomatrix(np.array(gray), [3], angles=[0], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_3_0=(contrast[0][0])
    energy_3_0=(energy[0][0])
    homogeneity_3_0=(homogeneity[0][0])
    correlation_3_0=(correlation[0][0])
        
    # distance 3 and angle 90
        
    glcm = graycomatrix(np.array(gray), [3], angles=[np.pi/2], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_3_90=(contrast[0][0])
    energy_3_90=(energy[0][0])
    homogeneity_3_90=(homogeneity[0][0])
    correlation_3_90=(correlation[0][0])
        
    # distance 5 and angle 0
        
    glcm = graycomatrix(np.array(gray), [5], angles=[0], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_5_0=(contrast[0][0])
    energy_5_0=(energy[0][0])
    homogeneity_5_0=(homogeneity[0][0])
    correlation_5_0=(correlation[0][0])
        
    # distance 5 and angle 90
        
    glcm = graycomatrix(np.array(gray), [5], angles=[np.pi/2], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_5_90=(contrast[0][0])
    energy_5_90=(energy[0][0])
    homogeneity_5_90=(homogeneity[0][0])
    correlation_5_90=(correlation[0][0])

    # distance 5 and angle 45
        
    glcm = graycomatrix(np.array(gray), [5], angles=[np.pi/4], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_5_45=(contrast[0][0])
    energy_5_45=(energy[0][0])
    homogeneity_5_45=(homogeneity[0][0])
    correlation_5_45=(correlation[0][0])

    # distance 3 and angle 45
        
    glcm = graycomatrix(np.array(gray), [3], angles=[np.pi/4], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_3_45=(contrast[0][0])
    energy_3_45=(energy[0][0])
    homogeneity_3_45=(homogeneity[0][0])
    correlation_3_45=(correlation[0][0])

    # distance 1 and angle 45
        
    glcm = graycomatrix(np.array(gray), [1], angles=[np.pi/4], levels=256, symmetric=True, normed=True)
        
    contrast = graycoprops(glcm, prop="contrast")
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    correlation = graycoprops(glcm, prop='correlation')
        
    contrast_1_45=(contrast[0][0])
    energy_1_45=(energy[0][0])
    homogeneity_1_45=(homogeneity[0][0])
    correlation_1_45=(correlation[0][0])

    features = {'contrast_1_0': contrast_1_0, 'energy_1_0': energy_1_0, 'homogeneity_1_0': homogeneity_1_0, 'correlation_1_0': correlation_1_0,
		'contrast_1_90': contrast_1_90, 'energy_1_90': energy_1_90, 'homogeneity_1_90': homogeneity_1_90, 'correlation_1_90': correlation_1_90,
		'contrast_3_0': contrast_3_0, 'energy_3_0': energy_3_0, 'homogeneity_3_0': homogeneity_3_0, 'correlation_3_0': correlation_3_0,
		'contrast_3_90': contrast_3_90, 'energy_3_90': energy_3_90, 'homogeneity_3_90': homogeneity_3_90, 'correlation_3_90': correlation_3_90,		
                'contrast_5_0': contrast_5_0, 'energy_5_0': energy_5_0, 'homogeneity_5_0': homogeneity_5_0, 'correlation_5_0': correlation_5_0,
		'contrast_5_90': contrast_5_90, 'energy_5_90': energy_5_90, 'homogeneity_5_90': homogeneity_5_90, 'correlation_5_90': correlation_5_90,
		'contrast_1_45': contrast_1_45, 'energy_1_45': energy_1_45, 'homogeneity_1_45': homogeneity_1_45, 'correlation_1_45': correlation_1_45,
		'contrast_3_45': contrast_3_45, 'energy_3_45': energy_3_45, 'homogeneity_3_45': homogeneity_3_45, 'correlation_3_45': correlation_3_45,
		'contrast_5_45': contrast_5_45, 'energy_5_45': energy_5_45, 'homogeneity_5_45': homogeneity_5_45, 'correlation_5_45': correlation_5_45,
		}
    return features
      
if(img is not None):
	features_dict = get_features(img)
	features_list = list(features_dict.values())
	scaler = joblib.load("best_scaler_5.joblib")
	features = scaler.transform(np.array(features_list).reshape(1,-1))
	model = joblib.load("best_knn_5.joblib")
	pred = model.predict(np.array(features).reshape(1,-1))
	label = pred[0]
	col1,col2,col3 = st.columns([2,1,2])
	with col2:
		# st.write(f"{round(model.predict_proba(np.array(features).reshape(1,-1)), 2)}")
		st.image(img)
		st.markdown("##### "+op[label])