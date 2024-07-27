import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if 'model' not in st.session_state:
    st.session_state.model = 'Brain Tumor Detection'
def update_radio():
    st.session_state.model =st.session_state.radio

if 'clas' not in st.session_state:
    st.session_state.clas = '15 Classes'
def update_selbox():
    st.session_state.clas =st.session_state.box

if 'check' not in st.session_state:
    st.session_state.check1 = False
def update_check():
    st.session_state.check1 =st.session_state.check

def update_photo():
    st.session_state.photo =st.session_state.image

def pred(img,radio,selbox,check):
    img = tf.keras.utils.load_img(
    img,
    grayscale=False,
    color_mode='rgb',
    target_size=(224,224),
    interpolation='nearest',
    keep_aspect_ratio=False
    )
    os.remove(st.session_state.image.name)
    img = np.array(img).reshape(-1, 224, 224, 3)
    if radio =='Alzheimer Detection':
        model = keras.models.load_model('alzheimer_99.5.h5')
        result=['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
    else:
        if selbox == '44 Classes':
            model = keras.models.load_model('44class_96.5.h5')
            result=['Astrocitoma T1','Astrocitoma T1C+','Astrocitoma T2','Carcinoma T1','Carcinoma T1C+','Carcinoma T2','Ependimoma T1','Ependimoma T1C+','Ependimoma T2','Ganglioglioma T1','Ganglioglioma T1C+',
            'Ganglioglioma T2','Germinoma T1','Germinoma T1C+','Germinoma T2','Glioblastoma T1','Glioblastoma T1C+','Glioblastoma T2','Granuloma T1','Granuloma T1C+','Granuloma T2','Meduloblastoma T1',
            'Meduloblastoma T1C+','Meduloblastoma T2','Meningioma T1','Meningioma T1C+','Meningioma T2','Neurocitoma T1','Neurocitoma T1C+','Neurocitoma T2','Oligodendroglioma T1','Oligodendroglioma T1C+',
            'Oligodendroglioma T2','Papiloma T1','Papiloma T1C+','Papiloma T2','Schwannoma T1','Schwannoma T1C+','Schwannoma T2','Tuberculoma T1','Tuberculoma T1C+','Tuberculoma T2','_NORMAL T1','_NORMAL T2']
        if selbox == '17 Classes':
            model = keras.models.load_model('17class_98.1.h5')
            result=['Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1','Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T1C+','Glioma (Astrocitoma, Ganglioglioma, Glioblastoma, Oligodendroglioma, Ependimoma) T2',
            'Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1','Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T1C+','Meningioma (de Baixo Grau, Atípico, Anaplásico, Transicional) T2','NORMAL T1','NORMAL T2','Neurocitoma (Central - Intraventricular, Extraventricular) T1','Neurocitoma (Central - Intraventricular, Extraventricular) T1C+',
            'Neurocitoma (Central - Intraventricular, Extraventricular) T2','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T1C+','Outros Tipos de Lesões (Abscessos, Cistos, Encefalopatias Diversas) T2','Schwannoma (Acustico, Vestibular - Trigeminal) T1',
            'Schwannoma (Acustico, Vestibular - Trigeminal) T1C+','Schwannoma (Acustico, Vestibular - Trigeminal) T2']
        if selbox == '15 Classes':
            model = keras.models.load_model('15class_99.8.h5')
            result=['Astrocitoma','Carcinoma','Ependimoma','Ganglioglioma','Germinoma','Glioblastoma','Granuloma','Meduloblastoma','Meningioma','Neurocitoma','Oligodendroglioma','Papiloma','Schwannoma','Tuberculoma','_NORMAL']
        if selbox == '2 Classes':
            model = keras.models.load_model('2calss_lagre_dataset_99.1.h5')
            result=['no', 'yes']
    pred= model.predict(img)
    if check:
        pred=pd.DataFrame({
        'class_name' : result,
        'pred_score' : pred.flatten()*100
        })
        pred.sort_values(['pred_score'],ascending = False,kind='stable',inplace=True)
        pred.reset_index(drop=True,inplace=True)
        return pred
    pred = np.argmax(pred, axis=1)
    return result[pred[0]]

def spr_sidebar():
    menu=option_menu(
        menu_title=None,
        options=['Home','About'],
        icons=['house','info-square'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )
    if menu=='Home':
        st.session_state.app_mode = 'Home'
    elif menu=='About':
        st.session_state.app_mode = 'About'
    
def home_page():
    st.session_state.check=st.session_state.check1
    st.session_state.radio=st.session_state.model
    st.session_state.box=st.session_state.clas
    if 'photo' in st.session_state:
        st.session_state.image=st.session_state.photo

    st.title('Brain MRI Tumor and Alzheimer Classification Web App')
    st.session_state.image=st.file_uploader('Upload MRI Image',accept_multiple_files=False,type=['png', 'jpg','jpeg'],key="upload",on_change=update_photo)
    if st.session_state.image != None:
        st.image(st.session_state.image,width=150)
        col,col2=st.columns([2,3])
        radio=col.radio("Model",options=('Brain Tumor Detection','Alzheimer Detection'),key='radio',on_change=update_radio)
        check=col.checkbox('Show Prediction Scores',key='check',on_change=update_check)
        if radio =='Brain Tumor Detection':
            selbox=col2.selectbox("choose a number of Classes",options=('44 Classes','17 Classes' ,'15 Classes','2 Classes'),index=0,key='box',on_change=update_selbox)
        else:
            selbox=col2.radio("choose a number of Classes",options=(['4 Classes']),index=0,key='box1',on_change=update_selbox)
        
        state =col.button('Get Result')
        if state:
            f=open(st.session_state.image.name, 'wb') 
            f.write(st.session_state.image.getbuffer())
            f.close()
            
            with st.spinner('Model Running....'):
                res=pred(st.session_state.image.name,radio,selbox,check)
            if check:
                col2.write(res)
            else :
                col2.success(str(res))
                
                    
                



def About_page():
   
    st.header('Data')
    """
    For the main model, I used [Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) a collection of T1, contrast-enhanced T1, and T2 magnetic resonance images separated by brain tumor type. Contains a total of 4479 images and 44 classes.
    
    I used this dataset to train my main CNN model and then tested it on different datasets. I used the same model and weights as the main model, with the only difference being the output layer. 
    ### Testing datasets 
    - [Brain Tumor MRI Images 44 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c) using only tumor types 4479 images and 15 classes
    - [Brain Tumor MRI Images 17 Classes](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-17-classes) contains 4448 images and 17 classes
    - [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) contains 3264 images and 4 classes
    - [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)contains 253 images and 2 classes
    - [Brain_Tumor_Detection_MRI](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri) contains 3060 images and 2 classes
    - [Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) contains 6400 images and 2 classes
    """

    """
    ## Made By-
    - Ameyaw Samuel Arthur [Linkedin](www.linkedin.com/in/samuel-arthur-ameyaw)
   
    """

def main():
    spr_sidebar()        
    if st.session_state.app_mode == 'Home':
        home_page()
    if st.session_state.app_mode == 'About' :
        About_page()
# Run main()
if __name__ == '__main__':
    main()

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)