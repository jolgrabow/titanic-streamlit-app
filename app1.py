# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()
# import znanych nam bibliotek

filename = "model.sv"
model = pickle.load(open(filename,'rb'))
# otwieramy wcześniej wytrenowany model

pclass_d = {0:"First",1:"Second", 2:"Third"}
embarked_d = {0:"Cherbourg", 1:"Queenstown", 2:"Southampton"}
sex_d = {0:"Female",1:"Male"}
# o ile wcześniej kodowaliśmy nasze zmienne, to teraz wprowadzamy etykiety z ich nazewnictwem

def main():

    st.set_page_config(page_title="Would you sink along with the Titanic?")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://media.istockphoto.com/photos/struggle-in-the-sea-picture-id1034893808")

    with overview:
        st.title("Would you sink along with the Titanic?")

    with left:
        sex_radio = st.radio( "Sex", list(sex_d.keys()), format_func=lambda x : sex_d[x] )
        embarked_radio = st.radio( "Port of embarkation", list(embarked_d.keys()), index=2, format_func= lambda x: embarked_d[x] )
        pclass_radio = st.radio( "Passenger class", list(pclass_d.keys()), index=2, format_func= lambda x: pclass_d[x] )

    with right:
        age_slider = st.slider("Age", value=1, min_value=1, max_value=76)
        sibsp_slider = st.slider("Amount of siblings + partner", min_value=0, max_value=8)
        parch_slider = st.slider("Amount of parents + children", min_value=0, max_value=9)
        fare_slider = st.slider("Ticket price", min_value=0, max_value=512, step=1)

    data = [[pclass_radio, sex_radio,  age_slider, sibsp_slider, parch_slider, fare_slider, embarked_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Would this person survive the catastrophe?")
        st.subheader(("Yes" if survival[0] == 1 else "No"))
        st.write("Prediction confidence {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
