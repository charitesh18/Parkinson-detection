import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
# from streamlit_option_menu import option_menu


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
#            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

parkinsons_data = pd.read_csv('/content/parkinsons.csv')
X = parkinsons_data.drop(columns=['name','status'], axis=1)
Y = parkinsons_data['status']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)



    
    # page title
st.title("Parkinson Disease Prediction")



    # code for Prediction
parkinsons_diagnosis = ''



def make_predictions(classifier, inputs):

    # Example input formatting (convert to numpy array)
  inputs_array = np.array(inputs).reshape(1, -1)
    # Perform prediction
  prediction = classifier.predict(inputs_array)
  return prediction
  
def get_user_inputs():
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
      input1 = st.number_input('Fo(Hz)')

    with col2:
      input2 = st.number_input('Fhi(Hz)')

    with col3:
      input3 = st.number_input('Flo(Hz)')

    with col4:
      input4 = st.number_input('Jitter(%)')

    with col5:
      input5 = st.number_input('Jitter(Abs)')

    with col1:
      input6 = st.number_input('RAP')

    with col2:
      input7 = st.number_input('PPQ')

    with col3:
      input8 = st.number_input('DDP')

    with col4:
      input9 = st.number_input('Shimmer')

    with col5:
      input10 = st.number_input('Shimmer(dB)')

    with col1:
      input11 = st.number_input('APQ3')

    with col2:
      input12 = st.number_input('APQ5')

    with col3:
      input13 = st.number_input('APQ')

    with col4:
      input14 = st.number_input('DDA')

    with col5:
      input15 = st.number_input('NHR')

    with col1:
      input16 = st.number_input('HNR')

    with col2:
      input17 = st.number_input('RPDE')

    with col3:
      input18 = st.number_input('DFA')

    with col4:
      input19 = st.number_input('spread1')

    with col5:
      input20 = st.number_input('spread2')

    with col1:
      input21 = st.number_input('D2')

    with col2:
      input22 = st.number_input('PPE')
    return input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20, input21, input22

# def main():

    # Title of the app
input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20, input21, input22 = get_user_inputs()
 
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
  
result = make_predictions(model, [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20, input21, input22])
if result == 1:
    parkinsons_diagnosis = "The person has the parkinson disease"
else:
    parkinsons_diagnosis = "The person does not have signs of the parkinson disease"
button_clicked = st.button('Predict')

# Check if the button is clicked
if button_clicked:
    st.write(parkinsons_diagnosis)




# st.success(parkinsons_diagnosis)
