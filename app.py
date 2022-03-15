import streamlit as st
import tensorflow as tf
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/waste_segregator.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # WASTE SEGREGATION USING AI
           by V A SAIRAM AND M MADHAV
         """
         )
st.text("Waste segregation is one of the important environmental aspects to be covered of.")
st.text("Mixing of these wastes can make their disposal difficult and can pose serious environmental issues.")
st.text("Generally, big hospials and healthcare centers have their own collection sites, incinerators, shredders etc.")
st.text("But most of these facilities are not seen in small-time hospitals, private laboratories and household.")
st.text("So proper care and awareness about waste disposal has to be addressed to these sectors.")
st.text("This project is used to categorise the given waste into one of the four prominent medical wastes and also provides an efficient and proper way to dispose them.")
file = st.file_uploader("Please upload any image from the local machine in case of computer or upload camera image in case of mobile", type=["jpg", "png","jpeg","webp"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        return prediction
if file is None:
    st.text("Please upload an image file within the allotted file size or retry of the error still persists...")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names= ['CHEMICAL:DISPOSE IT IN THE YELLOW BIN. This includes of expired and discarded chemicals, mercury from broken thermometer and used up or leaking batteries. It is important of not to store these wastes along with the normal wastes. There are special containers available for storing the waste. They are made with a material compatible for chemicals and are leak-proof. The main purpose is to avoid the chemical reaction between wastes. Every hospitals have their own collection sites. There are special collection sites for the household chemical waste. It is better to collect the waste from residents of a community and then sending them to the collection sites for proper disposal.', 
                  'GENERAL: DISPOSE IT IN THE GREEN BIN. There are no hazards with this type of waste and this can be disposed of on the normal waste bin which will be sent to burial if bio-degradable or sent to other places based on the materials.',
                  'INFECTIOUS: DISPOSE IT IN THE RED BIN. Most of the wastes contaning human-fluid falls into this category. It is better to collect all the wastes in one container and then sending them to the collection sites. An hospital may have their own collection site. There are waste collection sites for residential communities. This will be sent to an on-site steriliser and either to an incinerator or medical waste shredder based on the material involved in the waste.',
                  'SHARP: DISPOSE IT IN THE WHITE BIN. There are chances that some of the infectious waste can fall into this category. The FDA recommends a two-step process for properly disposing of used needles and other sharps. STEP 1- collect all sharp waste in a seperate container. STEP 2- dispose them to the collection site. The collection site can be specific to a hospital or community. Courier of waste to them is also advisable. All of these waste are sent to the shredder and if they contain some human fluid then they are sterilised before shredding. The output is sent to industries for manufacturing new products.' 
                  ]
    string= "THE PARTICULAR WASTE CATEGORY IS: " + class_names[np.argmax(predictions)]
    st.success(string) 
    string1="Most of the wastes cannot be disposed at our homes. There are special collection sites for the disposal. Our work is to isolate the wastes into seperate containers which can be send to the collection sites for proper disposal."  
    string2="HOPE THIS HELPS!! THANK YOU FOR USING THIS PROJECT!!!"
    st.success(string1)
    st.success(string2)