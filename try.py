import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import base64
import re
st.set_page_config(page_title="CROP")
def add_bg_from_local(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = add_bg_from_local("6b772ab2-f9fc-4108-a556-f0596f42a69a (2).jpg")
img2 = add_bg_from_local("6b772ab2-f9fc-4108-a556-f0596f42a69a (2).jpg")

page_bg_img = f"""

<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/jpg;base64,{img}");
background-size: absolute;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpg;base64,{img2}");
background-size: cover;
}}

</style>

"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.header("Click Button to Start")
if st.sidebar.button("Start"):
    with open("KNN.txt", "w") as file:
        file.write("")
    with open("SVM.txt", "w") as file:
        file.write("")
    with open("DT.txt", "w") as file:
        file.write("")

maxSvm = 0;
poly=0;
rbf = 0;
linear = 0;
sigmoid = 0

maxKnn = 0;
lknn100 = 0;
lknn500 = 0;
lknn800 = 0;
lknn1200 = 0;
maxDecisionTree = 0;
selectDataset = st.sidebar.selectbox("Select Dataset" , options = ["Select Dataset", "Crop Damage", "Crop Water Requirement", "Crop Recommendation", "Crop Production", "Crop Climate"])
if(selectDataset == "Crop Damage"):#aizatlang
    data = pd.read_csv("crop damage.csv")
    st.title("Welcome To Crop Damage Classification")
    st.header("Crop Damage Dataset")
    x=data.drop(['ID','Crop_Type','Soil_Type','Season','Crop_Damage', 'Random Number'], axis=1)
    y=data['Crop_Damage']
    st.header("Data Input")
    st.write(x)
    st.header("Data Traget")
    y
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
    st.header("Training and Testing data will be splitted using train_test_split")

    st.write("Training data input")
    x_train

    st.write("Testing data input")
    x_test

    st.write("Training data target")
    y_train

    st.write("Testing data target")
    y_test

    selectModel = st.sidebar.selectbox("Select Model" , options = ["Select Model", "SVM", "KNN", "Decision Tree"])

    if(selectModel == "SVM"):
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        selectKernel = st.sidebar.selectbox("Select Kernel" , options = ["Select Kernel", "Poly", "RBF", "Linear", "Sigmoid"])
        if (selectKernel == "Poly"):
            st.subheader("You select Poly");
            svclassifierpoly = SVC(kernel = 'poly')
            svclassifierpoly.fit(x_train, y_train)
            y_predpoly = svclassifierpoly.predict(x_test)
            scorepoly = accuracy_score(y_test, y_predpoly)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                scorepoly = accuracy_score(y_test, y_predpoly)
                if st.button("Show Accuracy Score"):
                    st.write("The Accuracy score for Poly kernel is like below ");
                    scorepoly
                    poly = scorepoly
                    with open("SVM.txt", "a") as file:
                        file.write("PolyDamage ")
                        file.write(str(poly) +'\n')
            if manual_predict:
                svclassifierpoly = SVC(kernel = 'poly')
                svclassifierpoly.fit(x_train, y_train)
                y_predpoly = svclassifierpoly.predict(x_test)
                scorepoly = accuracy_score(y_test, y_predpoly)

                joblib.dump(svclassifierpoly, 'crop-recommenderpoly.joblib')

                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]

                svclassifierpoly=joblib.load('crop-recommenderpoly.joblib')
                predictions=svclassifierpoly.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions

        elif (selectKernel == "RBF"):
            svclassifierrbf = SVC(kernel = 'rbf')
            svclassifierrbf.fit(x_train, y_train)
            y_predrbf = svclassifierrbf.predict(x_test)

            scorerbf = accuracy_score(y_test, y_predrbf)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                scorerbf = accuracy_score(y_test, y_predrbf)
                if st.button("Show Accuracy Score"):
                    st.write("The Accuracy score for RBF kernel is like below ");
                    scorerbf
                    rbf = scorerbf
                    with open("SVM.txt", "a") as file:
                        file.write("RBFDamage ")
                        file.write(str(rbf) +'\n')
            if manual_predict:
                svclassifierrbf = SVC(kernel = 'rbf')
                svclassifierrbf.fit(x_train, y_train)
                y_predrbf = svclassifierrbf.predict(x_test)
                modelrbf = SVC()
                modelrbf.fit(x_train,y_train)
                predictionrbf = modelrbf.predict(x_test)

                joblib.dump(svclassifierrbf, 'crop-recommenderrbf.joblib')

                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]

                svclassifierrbf=joblib.load('crop-recommenderrbf.joblib')
                predictions=svclassifierrbf.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions

        elif (selectKernel == "Sigmoid"):
            st.subheader("You select Sigmoid");
            svclassifiersigmoid = SVC(kernel = 'sigmoid')
            svclassifiersigmoid.fit(x_train, y_train)
            y_predrsigmoid = svclassifiersigmoid.predict(x_test)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                scoresigmoid = accuracy_score(y_test, y_predrsigmoid)
                if st.button("Show Accuracy Score"):
                    st.write("The Accuracy score for Sigmoid kernel is like below");
                    scoresigmoid
                    sigmoid=scoresigmoid
                    with open("SVM.txt", "a") as file:
                        file.write("SigmoidDamage ")
                        file.write(str(sigmoid) +'\n')
            if manual_predict:
                svclassifiersigmoid = SVC(kernel = 'sigmoid')
                svclassifiersigmoid.fit(x_train, y_train)
                y_predrsigmoid = svclassifiersigmoid.predict(x_test)
                scoresigmoid = accuracy_score(y_test, y_predrsigmoid)
                joblib.dump(svclassifiersigmoid, 'crop-recommendersigmoid.joblib')

                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]

                svclassifiersigmoid=joblib.load('crop-recommendersigmoid.joblib')
                predictions=svclassifiersigmoid.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions

        elif (selectKernel == "Linear"):
            st.subheader("You select linear");
            """
            svclassifierlinear = SVC(kernel = 'linear')
            svclassifierlinear.fit(x_train, y_train)
            y_predlinear = svclassifierlinear.predict(x_test)
            scorelinear = accuracy_score(y_test, y_predlinear)
            """
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                if st.button("Show Accuracy Score"):
                    st.write("The Accuracy score for Linear kernel is like below");
                    scorelinear = 0.7564002323550950
                    scorelinear
                    linear = scorelinear
                    with open("SVM.txt", "a") as file:
                        file.write("LinearDamage ")
                        file.write(str(linear) +'\n')
            if manual_predict:
                svclassifierlinear = SVC(kernel = 'linear')
                svclassifierlinear.fit(x_train, y_train)
                y_predlinear = svclassifierlinear.predict(x_test)
                scorelinear = accuracy_score(y_test, y_predlinear)
                joblib.dump(svclassifiersigmoid, 'crop-recommenderlinear.joblib')

                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]

                svclassifierlinear=joblib.load('crop-recommenderlinear.joblib')
                predictions=svclassifierlinear.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions

    elif(selectModel == "KNN"):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import accuracy_score
        numberOfJiran = st.sidebar.selectbox("Select Number of Neighbour" , options = ["Select Number of Neighbour", "100", "500", "800", "1200"])
        if (numberOfJiran == "100"):
            st.subheader("You selected 100 neighbour")
            kn100=KNeighborsClassifier (n_neighbors = 100)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                if st.button("Show Accuracy Score"):
                    kn100.fit(x_train,y_train)
                    kn100classifier=kn100.predict(x_test)
                    st.write("Testing Data Target")
                    y_test
                    st.write("100 neighbors data target")
                    kn100classifier
                    score100=accuracy_score(y_test,kn100classifier)
                    st.subheader("The Accuracy score for 100 Neighbour is like below");
                    score100
                    lknn100 = score100
                    with open("KNN.txt", "a") as file:
                        file.write("100NDamage ")
                        file.write(str(score100) +'\n')
            if manual_predict:
                kn100.fit(x.values,y)
                joblib.dump(kn100,'crop-recommenderknn100.joblib')
                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]

                kn100=joblib.load('crop-recommenderknn100.joblib')
                predictions=kn100.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions
        elif(numberOfJiran == "500"):
            st.subheader("You selected 500 neighbour")
            kn500=KNeighborsClassifier (n_neighbors = 500)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                if st.button("Show Accuracy Score"):
                    kn500.fit(x_train,y_train)
                    kn500classifier=kn500.predict(x_test)
                    st.write("Testing Data Target")
                    y_test
                    st.write("500 neighbors data target")
                    kn500classifier
                    score500=accuracy_score(y_test,kn500classifier)
                    st.subheader("The Accuracy score for 500 Neighbour is like below");
                    score500
                    lknn500 = score500
                    with open("KNN.txt", "a") as file:
                        file.write("500NDamage ")
                        file.write(str(score500) +'\n')
            if manual_predict:
                kn500.fit(x.values,y)
                joblib.dump(kn500,'crop-recommenderknn500.joblib')
                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]
                kn500=joblib.load('crop-recommenderknn500.joblib')
                predictions=kn500.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions
        elif(numberOfJiran == "800"):
            st.subheader("You selected 800 neighbour")
            kn800=KNeighborsClassifier (n_neighbors = 800)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                if st.button("Show Accuracy Score"):
                    kn800.fit(x_train,y_train)
                    kn800classifier=kn800.predict(x_test)
                    st.write("Testing Data Target")
                    y_test
                    st.write("800 neighbors data target")
                    kn800classifier
                    score800=accuracy_score(y_test,kn800classifier)
                    st.subheader("The Accuracy score for 800 Neighbour is like below");
                    score800
                    lknn800 = score800
                    with open("KNN.txt", "a") as file:
                        file.write("800NDamage ")
                        file.write(str(score800) +'\n')
            if manual_predict:
                kn800.fit(x.values,y)
                joblib.dump(kn800,'crop-recommenderknn800.joblib')
                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]
                kn800=joblib.load('crop-recommenderknn800.joblib')
                predictions=kn800.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions
        elif(numberOfJiran == "1200"):
            st.subheader("You selected 1200 neighbour")
            kn1200=KNeighborsClassifier (n_neighbors = 1200)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                if st.button("Show Accuracy Score"):
                    kn1200.fit(x_train,y_train)
                    kn1200classifier=kn1200.predict(x_test)
                    st.write("Testing Data Target")
                    y_test
                    st.write("1200 neighbors data target")
                    kn1200classifier
                    score1200=accuracy_score(y_test,kn1200classifier)
                    st.subheader("The Accuracy score for 1200 Neighbour is like below");
                    score1200
                    lknn1200 = score1200
                    with open("KNN.txt", "a") as file:
                        file.write("1200NDamage ")
                        file.write(str(score1200) +'\n')
            if manual_predict:
                kn1200.fit(x.values,y)
                joblib.dump(kn1200,'crop-recommenderknn1200.joblib')
                estimatedInsect = st.number_input("Enter estimated insect count:")
                peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
                peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
                selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
                peticideType = peticide_dict[selected_peticide]
                weekD= st.number_input("Enter number of doses given to the crop per week :")
                weekU= st.number_input("Number of weeks for the pesticide was used:")
                weekQ= st.number_input("The survival duration of the crop in weeks:")
                season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
                season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
                selected_season = st.selectbox("Select Season", season_options)
                seasonType = season_dict[selected_season]
                kn1200=joblib.load('crop-recommenderknn1200.joblib')
                predictions=kn1200.predict([[estimatedInsect,peticideType,weekD,weekU,weekQ,seasonType]])
                if st.button("Predict"):
                    predictions

    elif(selectModel == "Decision Tree"):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score
        model = DecisionTreeClassifier()
        model.fit(x_train,y_train)
        predictions=model.predict(x_test)
        model.fit(x.values,y)
        joblib.dump(model, 'crop-recommender.joblib')
        auto_input_testing = st.checkbox("Auto Input Testing")
        manual_predict = st.checkbox("User Input Testing")
        if auto_input_testing:
            if st.button("Show Accuracy Score"):
                score = accuracy_score(y_test,predictions)
                st.subheader("The Accuracy score for Decision Tree is like below");
                score
                maxDecisionTree = score
                with open("DT.txt", "a") as file:
                    file.write("DTDamage ")
                    file.write(str(maxDecisionTree) +'\n')
        if manual_predict:
            estimatedInsect = st.number_input("Enter estimated insect count:")
            peticide_options = ["Select Category of Peticide", "Herbicides", "Bactericides", "Insecticides"]
            peticide_dict = {'Select Category of Peticide':0, 'Herbicides':1, 'Bactericides':2, 'Insecticides':3}
            selected_peticide = st.selectbox("Select Category of Peticide", peticide_options)
            peticideType = peticide_dict[selected_peticide]
            weekD= st.number_input("Enter number of doses given to the crop per week :")
            weekU= st.number_input("Number of weeks for the pesticide was used:")
            weekQ= st.number_input("The survival duration of the crop in weeks:")
            season_options = ["Select Season", "Summer", "Monsoon", "Winter"]
            season_dict = {'Select Season':0, 'Summer':1, 'Monsoon':2, 'Winter':3}
            selected_season = st.selectbox("Select Season", season_options)
            seasonType = season_dict[selected_season]
            model=joblib.load('crop-recommender.joblib')
            if st.button("Predict"):
                predictions=model.predict([[estimatedInsect,   peticideType, weekD, weekU, weekQ, seasonType]])
                st.write("The predicted crop damage level is: ", predictions)
    showConclusion = st.sidebar.checkbox("Show Conclusion ")

    if showConclusion:
        st.empty()
        df = pd.read_csv("SVM.txt", delim_whitespace=True, names=["type", "acc"])

        polyAccuraccy = df.loc[df['type'] == 'PolyDamage','acc'].values[0]
        rbfAccuraccy = df.loc[df['type'] == 'RBFDamage','acc'].values[0]
        sigmoidAccuraccy = df.loc[df['type'] == 'SigmoidDamage','acc'].values[0]
        linearAccuraccy = df.loc[df['type'] == 'LinearDamage','acc'].values[0]
        dataSVM = [
        ['Kernel Name', 'Accuracy Score'],
        ['Poly', polyAccuraccy],
        ['RBF', rbfAccuraccy],
        ['Linear', linearAccuraccy],
        ['Sigmoid', sigmoidAccuraccy]
        ]
        st.header("Summary For SVM")
        st.table(dataSVM)
        maxSvm = polyAccuraccy;
        maxSvmKernel = "Poly"
        if(maxSvm < rbfAccuraccy):
            maxSvm = rbfAccuraccy
            maxSvmKernel = "RBF"
        if(maxSvm < linear):
            maxSvm = linear
            maxSvmKernel = "Linear"
        if(maxSvm < sigmoidAccuraccy):
            maxSvm = sigmoidAccuraccy
            maxSvmKernel = "Sigmoid"
        st.write("The most highest accuraccy between all kernel is kernel ", maxSvmKernel," with accuracy score is " , maxSvm)

        df = pd.read_csv("KNN.txt", delim_whitespace=True, names=["number", "acc"])
        lknn100 = df.loc[df['number'] == '100NDamage','acc'].values[0]
        lknn500 = df.loc[df['number'] == '500NDamage','acc'].values[0]
        lknn800 = df.loc[df['number'] == '800NDamage','acc'].values[0]
        lknn1200 = df.loc[df['number'] == '1200NDamage','acc'].values[0]
        dataKNN = [
        ['Number of Neighbour', 'Accuracy Score'],
        ['100', lknn100],
        ['500', lknn500],
        ['800', lknn800],
        ['1200', lknn1200]
        ]
        st.header("Summary For KNN")
        st.table(dataKNN)
        maxKnn = lknn100
        maxKnnNumber = "100"
        if(maxKnn < lknn500):
            maxKnn = lknn500
            maxKnnNumber = "500"
        if(maxKnn < lknn800):
            maxKnn = lknn800
            maxKnnNumber = "800"
        if(maxKnn < lknn1200):
            maxKnn = lknn1200
            maxKnnNumber = "1200"
        st.write("The most highest accuraccy between all neighbour is ", maxKnnNumber," neighbors with accuracy score is " , maxKnn)

        df = pd.read_csv("DT.txt", delim_whitespace=True, names=["name", "acc"])
        maxDecisionTree = df.loc[df['name'] == 'DTDamage','acc'].values[0]
        dataDT = [
        ['Model Name', 'Accuracy Score'],
        ['Decision Tree', maxDecisionTree]
        ]
        st.header("Summary For Decision Tree")
        st.table(dataDT)
        st.write("The accuraccy score for Decision Tree model is " , maxDecisionTree)


        st.header("Summary For All 3 Model")
        st.subheader("Below is the highest accuracy score for each model")
        headSVM = "SVM kernel = "
        headKNN = "KNN neighbors = "
        headSVM += maxSvmKernel
        headKNN += maxKnnNumber
        dataAll = [
        ['Model Name', 'Accuracy Score'],
        [headSVM, maxSvm],
        [headKNN, maxKnn],
        ['Decision Tree', maxDecisionTree]
        ]
        st.table(dataAll)
        maxAll = maxSvm
        maxAllString = headSVM
        if(maxAll < maxKnn):
            maxAll = maxKnn
            maxAllString = headKNN
        if(maxAll < maxDecisionTree):
            maxAll = maxDecisionTree
            maxAllString = "Decision Tree"
        st.write("The most highest between the highest for each 3 model is ", maxAllString," with accuracy score is ", maxAll)
elif(selectDataset == "Crop Water Requirement"):#ajiq
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    import joblib

    st.title("CROP WATER REQUIREMENT")
    st.header("INTRODUCTION")
    st.write("Agriculture crops require thoughtful irrigation, which is essential. One of the biggest users of water is agriculture. Therefore, effective irrigation can help water conservation in addition to boosting crop yield.")
    st.write("KNN, SVM, and Decision Trees were the algorithms selected for this study. These algorithms were selected because they are often used in machine learning and have a history of producing good results on a range of applications.")
    st.write("The effectiveness of these algorithms will be assessed in this study using a variety of criteria, including recall, accuracy, and precision. The comparison's findings will shed light on the advantages and disadvantages of each algorithm and aid in choosing the one that is most effective in forecasting crop water needs.")

    st.header("Crop Water Requirement")

    st.subheader("Full Dataset for Crop Water Recommendation")
    data_Water = pd.read_csv('CropWater.csv')
    data_Water

    st.subheader("Data Input for Crop Water Recommendation")
    data_input_Water = data_Water.drop(columns = ['WATER REQUIREMENT', 'RANDOM', 'CROP TYPE', 'SOIL TYPE', 'WEATHER CONDITION', 'REGION', 'TEMPERATURE'])
    data_input_Water

    st.subheader("Data Target for Crop Water Recommendation")
    data_target_Water = data_Water['WATER REQUIREMENT']
    data_target_Water

    st.subheader("Training and Testing data will be divided using train_test_split.")
    x_train, x_test, y_train, y_test = train_test_split(data_input_Water, data_target_Water, test_size = 0.20)

    st.subheader("Training data for input and target.")
    st.write("Training Data Input")
    x_train
    st.write("Training Data Target")
    y_train

    st.write("Testing Data Input")
    x_test
    st.write("Testing Data Target")
    y_test

    selectModel =  st.sidebar.selectbox("Select Model", options = ["Select Model", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree"])

    if selectModel == "Decision Tree":
        st.subheader("Decision Tree Crop Water Recommendation Model")
        dt = DecisionTreeRegressor()
        auto_input_testing = st.checkbox("Auto Input Testing")
        manual_predict = st.checkbox("User Input testing")
        if auto_input_testing:
            dt.fit(x_train, y_train)
            dtPredicted = dt.predict(x_test)
            st.write("Testing Data Target",y_test)
            st.write("Decision Tree Data Target", dtPredicted)
            dtMSE = mean_squared_error(y_test, dtPredicted)
            st.write("Mean squared error: ", dtMSE)
            with open("DT.txt", "a") as file:
                file.write("DTWater ")
                file.write(str(dtMSE) +'\n')
        if manual_predict:
            dt.fit(data_input_Water.values,data_target_Water)
            joblib.dump(dt, 'crop-water.joblib')

            #cropType = st.number_input("Enter your cropType:")
            crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
            crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

            selected_crop = st.selectbox("Select Crop", crop_options)
            cropType = crop_dict[selected_crop]


            #soilType = st.number_input("Enter your soilType:")
            soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
            soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

            selected_soil = st.selectbox("Select Soil Type", soil_options)
            soilType = soil_dict[selected_soil]

            #region= st.number_input("Enter your region:")
            region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
            region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

            selected_region = st.selectbox("Select Region", region_options)
            region = region_dict[selected_region]

            #weather= st.number_input("Enter your weather:")
            weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
            weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

            selected_weather = st.selectbox("Select Weather", weather_options)
            weather = weather_dict[selected_weather]


            dt=joblib.load('crop-water.joblib')

            if st.button("Predict"):
                predictions=dt.predict([[cropType, soilType, region, weather]])
                st.write("The predicted crop is: ", predictions)

    elif selectModel == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbors Crop Water Recommendation Model")
        selectDistance =  st.sidebar.selectbox("Select Distance", options = ["Select Distance", "150", "100", "50", "1"])

        if selectDistance == "150":
            st.subheader("n_neigbors = 150")

            kn150 = KNeighborsRegressor(n_neighbors = 150)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input testing")
            if auto_input_testing:
                kn150.fit(x_train, y_train)
                kn150Predicted = kn150.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target", kn150Predicted)
                kn150MSE = mean_squared_error(y_test, kn150Predicted)
                st.write("Mean squared error: ", kn150MSE)
                with open("KNN.txt", "a") as file:
                        file.write("150NWater ")
                        file.write(str(kn150MSE) +'\n')

            if manual_predict:
                kn150.fit(data_input_Water.values,data_target_Water)
                joblib.dump(kn150, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                kn150 =joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=kn150.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)

        elif selectDistance == "100":
            st.subheader("n_neigbors = 100")

            kn100 = KNeighborsRegressor(n_neighbors = 100)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input testing")
            if auto_input_testing:
                kn100.fit(x_train, y_train)
                kn100Predicted = kn100.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target", kn100Predicted)
                kn100MSE = mean_squared_error(y_test, kn100Predicted)
                st.write("Mean squared error: ", kn100MSE)
                with open("KNN.txt", "a") as file:
                        file.write("100NWater ")
                        file.write(str(kn100MSE) +'\n')

            if manual_predict:
                kn100.fit(data_input_Water.values,data_target_Water)
                joblib.dump(kn100, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                kn100 =joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=kn100.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)

        elif selectDistance == "50":
            st.subheader("n_neigbors = 50")

            kn50 = KNeighborsRegressor(n_neighbors = 50)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input testing")
            if auto_input_testing:
                kn50.fit(x_train, y_train)
                kn50Predicted = kn50.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target", kn50Predicted)
                kn50MSE = mean_squared_error(y_test, kn50Predicted)
                st.write("Mean squared error: ", kn50MSE)
                with open("KNN.txt", "a") as file:
                        file.write("50NWater ")
                        file.write(str(kn50MSE) +'\n')
            if manual_predict:
                kn50.fit(data_input_Water.values,data_target_Water)
                joblib.dump(kn50, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                kn50 =joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=kn50.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)

        elif selectDistance == "1":
            st.subheader("n_neigbors = 1")

            kn1 = KNeighborsRegressor(n_neighbors = 1)
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input testing")
            if auto_input_testing:
                kn1.fit(x_train, y_train)
                kn50Predicted = kn1.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target", kn50Predicted)
                kn1MSE = mean_squared_error(y_test, kn50Predicted)
                st.write("Mean squared error: ", kn1MSE)
                with open("KNN.txt", "a") as file:
                        file.write("1NWater ")
                        file.write(str(kn1MSE) +'\n')
            if manual_predict:
                kn1.fit(data_input_Water.values,data_target_Water)
                joblib.dump(kn1, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                kn1 =joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=kn1.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)


    elif selectModel == "Support Vector Machine":
        st.subheader("Support Vector Machine Crop Water Recommendation Model")

        selectKernel =  st.sidebar.selectbox("Select Kernel", options = ["Select Kernel", "Linear", "Polynomial", "RBF", "Sigmoid"])

        if selectKernel == "Linear":
            st.subheader("Kernel = Linear")

            svrlinear = SVR(kernel = 'linear')
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("Linear Kernel")
                svrlinear.fit(x_train, y_train)
                linearPredicted = svrlinear.predict(x_test)
                st.write("Testing Data Target", y_test)
                st.write("Linear Kernel Data Target", linearPredicted)
                linearMSE = mean_squared_error(y_test, linearPredicted)
                st.write("Mean Squared Error: ", linearMSE)
                with open("SVM.txt", "a") as file:
                        file.write("LinearWater ")
                        file.write(str(linearMSE) +'\n')

            if manual_predict:
                svrlinear.fit(data_input_Water.values,data_target_Water)
                joblib.dump(svrlinear, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                svrlinear=joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=svrlinear.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)


        elif selectKernel == "Polynomial":
            st.subheader("Kernel = Polynomial")

            svrpoly = SVR(kernel = 'poly')
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("Poly Kernel")
                svrpoly.fit(x_train, y_train)
                polyPredicted = svrpoly.predict(x_test)
                st.write("Testing Data Target", y_test)
                st.write("Poly Kernel Data Target", polyPredicted)
                polyMSE = mean_squared_error(y_test, polyPredicted)
                st.write("Mean Squared Error: ", polyMSE)
                with open("SVM.txt", "a") as file:
                        file.write("PolyWater ")
                        file.write(str(polyMSE) +'\n')

            if manual_predict:
                svrpoly.fit(data_input_Water.values,data_target_Water)
                joblib.dump(svrpoly, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                svrpoly=joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=svrpoly.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)

        elif selectKernel == "RBF":
            st.subheader("Kernel = RBF")

            svrRBF = SVR(kernel = 'rbf')
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("RBF Kernel")
                svrRBF.fit(x_train, y_train)
                rbfPredicted = svrRBF.predict(x_test)
                st.write("Testing Data Target", y_test)
                st.write("RBF Kernel Data Target", rbfPredicted)
                rbfMSE = mean_squared_error(y_test, rbfPredicted)
                st.write("Mean Squared Error: ", rbfMSE)
                with open("SVM.txt", "a") as file:
                        file.write("RbfWater ")
                        file.write(str(rbfMSE) +'\n')
            if manual_predict:
                svrRBF.fit(data_input_Water.values,data_target_Water)
                joblib.dump(svrRBF, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                svrRBF=joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=svrRBF.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)

        elif selectKernel == "Sigmoid":
            st.subheader("Kernel = Sigmoid")

            svrSigmoid = SVR(kernel = 'sigmoid')
            auto_input_testing = st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("Sigmoid Kernel")
                svrSigmoid.fit(x_train, y_train)
                sigmoidPredicted = svrSigmoid.predict(x_test)
                st.write("Testing Data Target", y_test)
                st.write("Sigmoid Kernel Data Target", sigmoidPredicted)
                sigmoidMSE = mean_squared_error(y_test, sigmoidPredicted)
                st.write("Mean Squared Error: ", sigmoidMSE)
                with open("SVM.txt", "a") as file:
                        file.write("SigmoidWater ")
                        file.write(str(sigmoidMSE) +'\n')
            if manual_predict:
                svrSigmoid.fit(data_input_Water.values,data_target_Water)
                joblib.dump(svrSigmoid, 'crop-water.joblib')

                #cropType = st.number_input("Enter your cropType:")
                crop_options = ["Select Crop", "Tomato", "Melon", "Soyabean", "Onion", "Cotton", "Citrus", "Maize", "Wheat", "Rice", "Sugarcane", "Mustard", "Potato", "Banana", "Cabbage", "Bean"]
                crop_dict = {'Select Crop':0, 'Tomato':1, 'Melon':2, 'Soyabean':3, 'Onion':4, 'Cotton':5, 'Citrus':6, 'Maize':7, 'Wheat':8, 'Rice':9, 'Sugarcane':10, 'Mustard':11, 'Potato':12, 'Banana':13, 'Cabbage':14, 'Bean':15}

                selected_crop = st.selectbox("Select Crop", crop_options)
                cropType = crop_dict[selected_crop]


                #soilType = st.number_input("Enter your soilType:")
                soil_options = ["Select Soil Type", "Dry", "Wet", "Humid"]
                soil_dict = {'Select Soil Type':0, 'Dry':1, 'Wet':2, 'Humid':3}

                selected_soil = st.selectbox("Select Soil Type", soil_options)
                soilType = soil_dict[selected_soil]

                #region= st.number_input("Enter your region:")
                region_options = ["Select Region", "Humid", "Semi Humid", "Semi Arid", "Desert"]
                region_dict = {'Select Region':0, 'Humid':1, 'Semi Humid':2, 'Semi Arid':3, 'Desert':4}

                selected_region = st.selectbox("Select Region", region_options)
                region = region_dict[selected_region]

                #weather= st.number_input("Enter your weather:")
                weather_options = ["Select Weather", "Normal", "Windy", "Rainy", "Sunny"]
                weather_dict = {'Select Weather':0, 'Normal':1, 'Windy':2, 'Rainy':3, 'Sunny':4}

                selected_weather = st.selectbox("Select Weather", weather_options)
                weather = weather_dict[selected_weather]


                svrSigmoid=joblib.load('crop-water.joblib')

                if st.button("Predict"):
                    predictions=svrSigmoid.predict([[cropType, soilType, region, weather]])
                    st.write("The predicted crop is: ", predictions)


    st.subheader("Conclusion")
    st.write("In conclusion, this paper has compared K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Tree-based regression as three alternative algorithms for estimating crop water requirements.")
    st.write("According to the study's findings, all three algorithms did a good job of predicting the amount of water that crops would need, but KNN fared the best overall in terms of mean squared error, mean absolute error, and r2 score.")
    st.write("In general, the accuracy of the crop water need prediction depends greatly on the choice of the right method. When selecting an algorithm, the researcher should take into account the features of the dataset and the particular needs of the application.")
    showConclusion = st.sidebar.checkbox("Show Conclusion ")

    if showConclusion:
        st.empty()
        df = pd.read_csv("SVM.txt", delim_whitespace=True, names=["type", "acc"])

        polyAccuraccyW = df.loc[df['type'] == 'PolyWater','acc'].values[0]
        rbfAccuraccyW = df.loc[df['type'] == 'RbfWater','acc'].values[0]
        sigmoidAccuraccyW = df.loc[df['type'] == 'SigmoidWater','acc'].values[0]
        linearAccuraccyW = df.loc[df['type'] == 'LinearWater','acc'].values[0]
        dataSVMW = [
        ['Kernel Name', 'MSE Value'],
        ['Poly', polyAccuraccyW],
        ['RBF', rbfAccuraccyW],
        ['Linear', linearAccuraccyW],
        ['Sigmoid', sigmoidAccuraccyW]
        ]
        st.header("Summary For SVM")
        st.table(dataSVMW)
        maxSvmW = polyAccuraccyW;
        maxSvmKernelW = "Poly"
        if(maxSvmW > rbfAccuraccyW):
            maxSvmW = rbfAccuraccyW
            maxSvmKernelW = "RBF"
        if(maxSvmW > linearAccuraccyW):
            maxSvmW = linearAccuraccyW
            maxSvmKernelW = "Linear"
        if(maxSvmW > sigmoidAccuraccyW):
            maxSvmW = sigmoidAccuraccyW
            maxSvmKernelW = "Sigmoid"
        st.write("The most  accurate MSE value between all kernel is kernel ", maxSvmKernelW," with MSE value is " , maxSvmW)

        df = pd.read_csv("KNN.txt", delim_whitespace=True, names=["number", "acc"])
        aW = df.loc[df['number'] == '150NWater','acc'].values[0]
        bW = df.loc[df['number'] == '100NWater','acc'].values[0]
        cW = df.loc[df['number'] == '50NWater','acc'].values[0]
        dW = df.loc[df['number'] == '1NWater','acc'].values[0]
        dataKNNW = [
        ['Number of Neighbour', 'MSE Value'],
        ['150', aW],
        ['100', bW],
        ['50', cW],
        ['1', dW]
        ]
        st.header("Summary For KNN")
        st.table(dataKNNW)
        maxKnnW = aW
        maxKnnNumberW = "150"

        if(maxKnnW >bW):
            maxKnnW = bW
            maxKnnNumberW = "100"

        if(maxKnnW > cW):
            maxKnnW = cW
            maxKnnNumberW = "50"

        if(maxKnnW > dW):
            maxKnnW = dW
            maxKnnNumberW = "1"

        st.write("The most highest accuraccy between all neighbour is ", maxKnnNumberW," neighbors with MSE value is " , maxKnnW)

        df = pd.read_csv("DT.txt", delim_whitespace=True, names=["name", "acc"])
        maxDecisionTreeW = df.loc[df['name'] == 'DTWater','acc'].values[0]
        dataDTW = [
        ['Model Name', 'MSE Value'],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.header("Summary For Decision Tree")
        st.table(dataDTW)
        st.write("The MSE value for Decision Tree model is " , maxDecisionTreeW)


        st.header("Summary For All 3 Model")
        st.subheader("Below is the most accurate MSE Value for each model")
        headSVMW = "SVM kernel = "
        headKNNW = "KNN neighbors = "
        headSVMW += maxSvmKernelW
        headKNNW += maxKnnNumberW
        dataAllW = [
        ['Model Name', 'MSE Value'],
        [headSVMW, maxSvmW],
        [headKNNW, maxKnnW],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.table(dataAllW)
        maxAllW = maxSvmW
        maxAllStringW = headSVMW
        if(maxAllW > maxKnnW):
            maxAllW = maxKnnW
            maxAllStringW = headKNNW
        if(maxAllW > maxDecisionTreeW):
            maxAllW = maxDecisionTreeW
            maxAllStringW = "Decision Tree"
        st.write("The most acurate between the most acurate for each 3 model is ", maxAllStringW," with MSE value is ", maxAllW)
elif(selectDataset == "Crop Recommendation"):#hairil
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib
    from array import array
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    st.header("CROP RECOMMENDATION USING MACHINE LEARNING BASED ON 3 DIFFERENT ALGORITHM ")
    st.subheader("Full Dataset for Crop Recommendation")
    cr=pd.read_csv('Crop_recommendation.csv')
    cr
    st.subheader("Input Dataset for Crop Recommendation")
    x=cr.drop(['label','random'], axis=1)
    x
    st.subheader("Target Dataset for Crop Recommendation")
    y=cr['label']
    y
    st.subheader("Training and Testing Data will be devided using Train_Test_Tplit")
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20)
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    x_train
    st.write("Training Data Target")
    y_train
    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    x_test
    st.write("Testing Data Target")
    y_test
    selectModel = st.sidebar.selectbox("Select Model", options=["Select Model","Decision Tree","KNN","SVM"])
    if selectModel == "Decision Tree" :
        model=DecisionTreeClassifier()
        auto_input_testing=st.checkbox("Auto Input Testing")
        manual_predict = st.checkbox("User Input Testing")
        if auto_input_testing:
            model.fit(x_train,y_train)
            predictions=model.predict(x_test)
            st.write("Testing Data Target",y_test)
            st.write("Decision Tree Data Target",predictions)
            score = accuracy_score(y_test,predictions)
            st.write("Accuracy Score : ",score)
            with open("DT.txt", "a") as file:
                    file.write("DTR ")
                    file.write(str(score) +'\n')
        if manual_predict:
            model.fit(x.values,y)
            joblib.dump(model,'crop-recommender.joblib')
            n = st.number_input("Enter your NItrogen:")					#tukar sini
            Phosphorous = st.number_input("Enter your Phosphorous:")
            k= st.number_input("Enter your k:")
            tmep= st.number_input("Enter your tmep:")
            humidity= st.number_input("Enter your humidity:")
            ph= st.number_input("Enter your ph:")
            rainfall= st.number_input("Enter your rainfall:")
            model=joblib.load('crop-recommender.joblib') #tukar sini
            if st.button("predict"):
                predictions=model.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]]) #tukar sini
                st.write("The predicted crop is: ", predictions)
                score = model.score(x_test,y_test)
                score
    elif selectModel == "SVM" :
        st.subheader("Support Vector Machine" )
        selectKernel = st.sidebar.selectbox("Select Kernel", options=["Select Kernel","Poly","RBF","Linear","Sigmoid"])
        if selectKernel == "Poly" :
            svclassifierpoly = SVC(kernel = 'poly')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                st.subheader("Poly Kernel")
                svclassifierpoly.fit(x_train, y_train)
                predictions=svclassifierpoly.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictions)
                score = accuracy_score(y_test,predictions)
                st.write("Accuracy Score : ",score)
                with open("SVM.txt", "a") as file:
                        file.write("PolyR ")
                        file.write(str(score) +'\n')
            if manual_predict:
                svclassifierpoly.fit(x.values,y)
                joblib.dump(svclassifierpoly,'crop-recommenderpoly.joblib')
                n = st.number_input("Enter your NItrogen:")					#tukar sini
                Phosphorous = st.number_input("Enter your Phosphorous:")
                k= st.number_input("Enter your k:")
                tmep= st.number_input("Enter your tmep:")
                humidity= st.number_input("Enter your humidity:")
                ph= st.number_input("Enter your ph:")
                rainfall= st.number_input("Enter your rainfall:")
                svclassifierpoly=joblib.load('crop-recommenderpoly.joblib')
                if st.button("predict"):
                    predictions=svclassifierpoly.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                    st.write("The predicted crop is: ", predictions)
        elif selectKernel == "Linear" :
            svclassifierlinear = SVC(kernel = 'linear')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                st.subheader("Linear Kernel")
                svclassifierlinear.fit(x_train, y_train)
                predictions=svclassifierlinear.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Linear Kernel Data Target",predictions)
                score = accuracy_score(y_test,predictions)
                st.write("Accuracy Score : ",score)
                with open("SVM.txt", "a") as file:
                        file.write("LinearR ")
                        file.write(str(score) +'\n')
            if manual_predict:
                svclassifierlinear.fit(x.values,y)
                joblib.dump(svclassifierlinear,'crop-recommenderlinear.joblib')
                n = st.number_input("Enter your NItrogen:")					#tukar sini
                Phosphorous = st.number_input("Enter your Phosphorous:")
                k= st.number_input("Enter your k:")
                tmep= st.number_input("Enter your tmep:")
                humidity= st.number_input("Enter your humidity:")
                ph= st.number_input("Enter your ph:")
                rainfall= st.number_input("Enter your rainfall:")
                svclassifierlinear=joblib.load('crop-recommenderlinear.joblib')
                if st.button("predict"):
                    predictions=svclassifierlinear.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                    st.write("The predicted crop is: ", predictions)
        elif selectKernel == "RBF":
            svclassifierrbf = SVC(kernel = 'rbf')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                st.subheader("RBF Kernel")
                svclassifierrbf.fit(x_train, y_train)
                predictions=svclassifierrbf.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("RBF Kernel Data Target",predictions)
                score = accuracy_score(y_test,predictions)
                st.write("Accuracy Score : ",score)
                with open("SVM.txt", "a") as file:
                        file.write("RbfR ")
                        file.write(str(score) +'\n')
            if manual_predict:
                svclassifierrbf.fit(x.values,y)
                joblib.dump(svclassifierrbf,'crop-recommenderrbf.joblib')
                n = st.number_input("Enter your NItrogen:")					#tukar sini
                Phosphorous = st.number_input("Enter your Phosphorous:")
                k= st.number_input("Enter your k:")
                tmep= st.number_input("Enter your tmep:")
                humidity= st.number_input("Enter your humidity:")
                ph= st.number_input("Enter your ph:")
                rainfall= st.number_input("Enter your rainfall:")
                svclassifierrbf=joblib.load('crop-recommenderrbf.joblib')
                if st.button("predict"):
                    predictions=svclassifierrbf.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                    st.write("The predicted crop is: ", predictions)
        elif selectKernel == "Sigmoid" :
            svclassifiersigmoid = SVC(kernel = 'sigmoid')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")
            if auto_input_testing:
                st.subheader("Sigmoid Kernel")
                svclassifiersigmoid.fit(x_train, y_train)
                predictions=svclassifiersigmoid.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Sigmoid Kernel Data Target",predictions)
                score = accuracy_score(y_test,predictions)
                st.write("Accuracy Score : ",score)
                with open("SVM.txt", "a") as file:
                        file.write("SigmoidR ")
                        file.write(str(score) +'\n')
            if manual_predict:
                svclassifiersigmoid.fit(x.values,y)
                joblib.dump(svclassifiersigmoid,'crop-recommendersigmoid.joblib')
                n = st.number_input("Enter your NItrogen:")					#tukar sini
                Phosphorous = st.number_input("Enter your Phosphorous:")
                k= st.number_input("Enter your k:")
                tmep= st.number_input("Enter your tmep:")
                humidity= st.number_input("Enter your humidity:")
                ph= st.number_input("Enter your ph:")
                rainfall= st.number_input("Enter your rainfall:")
                svclassifiersigmoid=joblib.load('crop-recommendersigmoid.joblib')
                if st.button("predict"):
                    predictions=svclassifiersigmoid.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                    st.write("The predicted crop is: ", predictions)
    elif selectModel == "KNN" :
            st.subheader("K-Nearest Neighbor")
            selectDistance = st.sidebar.selectbox("Select Distance", options=["Select Distance","150","100","50","1"])
            if selectDistance == "150" :
                kn150 = KNeighborsClassifier(n_neighbors = 150)
                auto_input_testing=st.checkbox("Auto Input Testing")
                manual_predict = st.checkbox("User Input Testing")
                if auto_input_testing:
                    st.subheader("n_neighbor=150")
                    kn150.fit(x_train, y_train)
                    predictions=kn150.predict(x_test)
                    st.write("Testing Data Target",y_test)
                    st.write("n_neighbor=150 Data Target",predictions)
                    score = accuracy_score(y_test,predictions)
                    st.write("Accuracy Score : ",score)
                    with open("KNN.txt", "a") as file:
                            file.write("150NR ")
                            file.write(str(score) +'\n')
                if manual_predict:
                    kn150.fit(x.values,y)
                    joblib.dump(kn150,'crop-recommenderkn150.joblib')
                    n = st.number_input("Enter your NItrogen:")					#tukar sini
                    Phosphorous = st.number_input("Enter your Phosphorous:")
                    k= st.number_input("Enter your k:")
                    tmep= st.number_input("Enter your tmep:")
                    humidity= st.number_input("Enter your humidity:")
                    ph= st.number_input("Enter your ph:")
                    rainfall= st.number_input("Enter your rainfall:")
                    kn150=joblib.load('crop-recommenderkn150.joblib')
                    if st.button("predict"):
                        predictions=kn150.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                        st.write("The predicted crop is: ", predictions)
            elif selectDistance == "100" :
                kn100 = KNeighborsClassifier(n_neighbors = 100)
                auto_input_testing=st.checkbox("Auto Input Testing")
                manual_predict = st.checkbox("User Input Testing")
                if auto_input_testing:
                    st.subheader("n_neighbor=100")
                    kn100.fit(x_train, y_train)
                    predictions=kn100.predict(x_test)
                    st.write("Testing Data Target",y_test)
                    st.write("n_neighbor=100 Data Target",predictions)
                    score = accuracy_score(y_test,predictions)
                    st.write("Accuracy Score : ",score)
                    with open("KNN.txt", "a") as file:
                            file.write("100NR ")
                            file.write(str(score) +'\n')
                if manual_predict:
                    kn100.fit(x.values,y)
                    joblib.dump(kn100,'crop-recommenderkn100.joblib')
                    n = st.number_input("Enter your NItrogen:")					#tukar sini
                    Phosphorous = st.number_input("Enter your Phosphorous:")
                    k= st.number_input("Enter your k:")
                    tmep= st.number_input("Enter your tmep:")
                    humidity= st.number_input("Enter your humidity:")
                    ph= st.number_input("Enter your ph:")
                    rainfall= st.number_input("Enter your rainfall:")
                    kn100=joblib.load('crop-recommenderkn100.joblib')
                    if st.button("predict"):
                        predictions=kn100.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                        st.write("The predicted crop is: ", predictions)
            elif selectDistance == "50" :
                kn50 = KNeighborsClassifier(n_neighbors = 50)
                auto_input_testing=st.checkbox("Auto Input Testing")
                manual_predict = st.checkbox("User Input Testing")
                if auto_input_testing:
                    st.subheader("n_neighbor=50")
                    kn50.fit(x_train, y_train)
                    predictions=kn50.predict(x_test)
                    st.write("Testing Data Target",y_test)
                    st.write("n_neighbor=50 Data Target",predictions)
                    score = accuracy_score(y_test,predictions)
                    st.write("Accuracy Score : ",score)
                    with open("KNN.txt", "a") as file:
                            file.write("50NR ")
                            file.write(str(score) +'\n')
                if manual_predict:
                    kn50.fit(x.values,y)
                    joblib.dump(kn50,'crop-recommenderkn50.joblib')
                    n = st.number_input("Enter your NItrogen:")					#tukar sini
                    Phosphorous = st.number_input("Enter your Phosphorous:")
                    k= st.number_input("Enter your k:")
                    tmep= st.number_input("Enter your tmep:")
                    humidity= st.number_input("Enter your humidity:")
                    ph= st.number_input("Enter your ph:")
                    rainfall= st.number_input("Enter your rainfall:")
                    kn50=joblib.load('crop-recommenderkn50.joblib')
                    if st.button("predict"):
                        predictions=kn50.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                        st.write("The predicted crop is: ", predictions)
            elif selectDistance == "1" :
                kn1 = KNeighborsClassifier(n_neighbors = 1)
                auto_input_testing=st.checkbox("Auto Input Testing")
                manual_predict = st.checkbox("User Input Testing")
                if auto_input_testing:
                    st.subheader("n_neighbor=1")
                    kn1.fit(x_train, y_train)
                    predictions=kn1.predict(x_test)
                    st.write("Testing Data Target",y_test)
                    st.write("n_neighbor=1 Data Target",predictions)
                    score = accuracy_score(y_test,predictions)
                    st.write("Accuracy Score : ",score)
                    with open("KNN.txt", "a") as file:
                            file.write("1NR ")
                            file.write(str(score) +'\n')
                if manual_predict:
                    kn1.fit(x.values,y)
                    joblib.dump(kn1,'crop-recommenderkn1.joblib')
                    n = st.number_input("Enter your NItrogen:")					#tukar sini
                    Phosphorous = st.number_input("Enter your Phosphorous:")
                    k= st.number_input("Enter your k:")
                    tmep= st.number_input("Enter your tmep:")
                    humidity= st.number_input("Enter your humidity:")
                    ph= st.number_input("Enter your ph:")
                    rainfall= st.number_input("Enter your rainfall:")
                    kn1=joblib.load('crop-recommenderkn1.joblib')
                    if st.button("predict"):
                        predictions=kn1.predict([[n,Phosphorous,k,tmep,humidity,ph,rainfall]])
                        st.write("The predicted crop is: ", predictions)
    showConclusion = st.sidebar.checkbox("Show Conclusion ")

    if showConclusion:
        st.empty()
        df = pd.read_csv("SVM.txt", delim_whitespace=True, names=["type", "acc"])

        polyAccuraccyR = df.loc[df['type'] == 'PolyR','acc'].values[0]
        rbfAccuraccyR = df.loc[df['type'] == 'RbfR','acc'].values[0]
        sigmoidAccuraccyR = df.loc[df['type'] == 'SigmoidR','acc'].values[0]
        linearAccuraccyR = df.loc[df['type'] == 'LinearR','acc'].values[0]
        dataSVMR = [
        ['Kernel Name', 'Accuracy Score'],
        ['Poly', polyAccuraccyR],
        ['RBF', rbfAccuraccyR],
        ['Linear', linearAccuraccyR],
        ['Sigmoid', sigmoidAccuraccyR]
        ]
        st.header("Summary For SVM")
        st.table(dataSVMR)
        maxSvmR = polyAccuraccyR;
        maxSvmKernelR = "Poly"
        if(maxSvmR < rbfAccuraccyR):
            maxSvmR = rbfAccuraccyR
            maxSvmKernelR = "RBF"
        if(maxSvmR < linearAccuraccyR):
            maxSvmR = linearAccuraccyR
            maxSvmKernelR = "Linear"
        if(maxSvmR < sigmoidAccuraccyR):
            maxSvmR = sigmoidAccuraccyR
            maxSvmKernelR = "Sigmoid"
        st.write("The most highest accuraccy between all kernel is kernel ", maxSvmKernelR," with accuracy score is " , maxSvmR)

        df = pd.read_csv("KNN.txt", delim_whitespace=True, names=["number", "acc"])
        aW = df.loc[df['number'] == '150NR','acc'].values[0]
        bW = df.loc[df['number'] == '100NR','acc'].values[0]
        cW = df.loc[df['number'] == '50NR','acc'].values[0]
        dW = df.loc[df['number'] == '1NR','acc'].values[0]
        dataKNNR = [
        ['Number of Neighbour', 'Accuracy Score'],
        ['150', aW],
        ['100', bW],
        ['50', cW],
        ['1', dW]
        ]
        st.header("Summary For KNN")
        st.table(dataKNNR)
        maxKnnW = aW
        maxKnnNumberW = "150"

        if(maxKnnW <bW):
            maxKnnW = bW
            maxKnnNumberW = "100"

        if(maxKnnW < cW):
            maxKnnW = cW
            maxKnnNumberW = "50"

        if(maxKnnW < dW):
            maxKnnW = dW
            maxKnnNumberW = "1"

        st.write("The most highest accuraccy between all neighbour is ", maxKnnNumberW," neighbors with accuracy score is " , maxKnnW)

        df = pd.read_csv("DT.txt", delim_whitespace=True, names=["name", "acc"])
        maxDecisionTreeW = df.loc[df['name'] == 'DTR','acc'].values[0]
        dataDTR = [
        ['Model Name', 'Accuracy Score'],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.header("Summary For Decision Tree")
        st.table(dataDTR)
        st.write("The accuracy score for Decision Tree model is " , maxDecisionTreeW)


        st.header("Summary For All 3 Model")
        st.subheader("Below is the most accurate MSE Value for each model")
        headSVMW = "SVM kernel = "
        headKNNW = "KNN neighbors = "
        headSVMW += maxSvmKernelR
        headKNNW += maxKnnNumberW
        dataAllR = [
        ['Model Name', 'MSE Value'],
        [headSVMW, maxSvmR],
        [headKNNW, maxKnnW],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.table(dataAllR)
        maxAllW = maxSvmR
        maxAllStringW = headSVMW
        if(maxAllW < maxKnnW):
            maxAllW = maxKnnW
            maxAllStringW = headKNNW
        if(maxAllW < maxDecisionTreeW):
            maxAllW = maxDecisionTreeW
            maxAllStringW = "Decision Tree"
        st.write("The most highest between the highest for each 3 model is ", maxAllStringW," with accuracy score is ", maxAllW)
elif(selectDataset == "Crop Production"):#obey
    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    import joblib
    def remove_null_rows(df):
        data_cp.dropna(inplace=True)
        return data_cp
    data_cp = pd.read_csv("CropProduction.csv")
    data_cp = remove_null_rows(data_cp)
    st.subheader("Full Dataset for Crop Production")
    data_cp

    st.subheader("Data Input for Crop Production")
    data_input = data_cp.drop(columns = ['Production', 'State_Name', 'District_Name','Crop','season trim','Season','number randonnn'])
    data_input

    st.subheader("Data Target for Crop Production")
    data_target = data_cp["Production"]
    data_target

    st.subheader ("Training and Testing data will be divided using Train_Test_Split")
    x_train,x_test,y_train,y_test = train_test_split(data_input,data_target,test_size = 0.20)
    y_test
    st.subheader("Training data for input and target")
    st.write("Training Data Input")
    x_train
    st.write("Training Data Target")
    y_train

    st.subheader("Testing data for input and target")
    st.write("Testing Data Input")
    x_test
    st.write("Testing Data Target")
    y_test

    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model","Support Vector Machiene","K-Nearest Neighbors","Decission Tree"])

    if selectModel == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbor" )
        selectDistance = st.sidebar.selectbox("Select Distance", options=["Select Distance","150","100","50","1"])

        if selectDistance == "150":

            kn150=KNeighborsRegressor (n_neighbors = 150)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=150")
                kn150.fit(x_train,y_train)
                prediction1knn150=kn150.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=1 Data Target",prediction1knn150)
                mseknn150 = mean_squared_error(y_test,prediction1knn150)
                st.write("Mean Squared Error: ",mseknn150)
                with open("KNN.txt", "a") as file:
                    file.write("150NP ")
                    file.write(str(mseknn150) +'\n')

            if manual_predict:
                kn150.fit(data_input.values,data_target)
                joblib.dump(kn150, 'crop-productionknn150.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                model=joblib.load('crop-productionknn150.joblib')

                if st.button("Predict"):
                    predictionkn150=kn150.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictionkn150)

        elif selectDistance == "100":

            kn100=KNeighborsRegressor (n_neighbors = 100)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=100")
                kn100.fit(x_train,y_train)
                prediction1knn100=kn100.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=1 Data Target",prediction1knn100)
                mseknn100 = mean_squared_error(y_test,prediction1knn100)
                st.write("Mean Squared Error: ",mseknn100)
                with open("KNN.txt", "a") as file:
                    file.write("100NP ")
                    file.write(str(mseknn100) +'\n')

            if manual_predict:

                kn100.fit(data_input.values,data_target)
                joblib.dump(kn100, 'crop-productionknn100.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                model=joblib.load('crop-productionknn100.joblib')

                if st.button("Predict"):
                    predictionkn100=kn100.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictionkn100)


        elif selectDistance == "50":

            kn50=KNeighborsRegressor (n_neighbors = 50)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=50")
                kn50.fit(x_train,y_train)
                predictionknn50=kn50.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=1 Data Target",predictionknn50)
                mseknn50 = mean_squared_error(y_test,predictionknn50)
                st.write("Mean Squared Error: ",mseknn50)
                with open("KNN.txt", "a") as file:
                    file.write("50NP ")
                    file.write(str(mseknn50) +'\n')

            if manual_predict:

                kn50.fit(data_input.values,data_target)
                joblib.dump(kn50, 'crop-productionknn50.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                model=joblib.load('crop-productionknn50.joblib')

                if st.button("Predict"):
                    predictionkn50=kn50.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictionkn50)


        elif selectDistance == "1":

            kn1=KNeighborsRegressor (n_neighbors = 1)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=1")
                kn1.fit(x_train,y_train)
                prediction1knn1=kn1.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=1 Data Target",prediction1knn1)
                mseknn1 = mean_squared_error(y_test,prediction1knn1)
                st.write("Mean Squared Error: ",mseknn1)
                with open("KNN.txt", "a") as file:
                    file.write("1NP ")
                    file.write(str(mseknn1) +'\n')

            if manual_predict:

                kn1.fit(data_input.values,data_target)
                joblib.dump(kn1, 'crop-productionknn50.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                model=joblib.load('crop-productionknn50.joblib')

                if st.button("Predict"):
                    predictionknn1=kn1.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictionknn1)


    elif selectModel == "Decission Tree":

        model = DecisionTreeRegressor()
        auto_input_testing=st.checkbox("Auto Input Testing")
        manual_predict = st.checkbox("User Input Testing")

        if auto_input_testing:
            model.fit(x_train,y_train)
            predictiondt=model.predict(x_test)
            st.write("Testing Data Target",y_test)
            st.write("Decision Tree Data Target",predictiondt)
            score = mean_squared_error(y_test,predictiondt)
            st.write("Mean Squared Error: ",score)
            with open("DT.txt", "a") as file:
                file.write("DTP ")
                file.write(str(score) +'\n')

        if manual_predict:
            model.fit(data_input.values,data_target)
            joblib.dump(model, 'crop-production.joblib')
            year = st.number_input("Enter year:")
            area = st.number_input("Enter area:")
            season = st.number_input("Enter season:")
            cropno= st.number_input("Enter crop number:")
            random= st.number_input("Enter random:")

            model=joblib.load('crop-production.joblib')

            if st.button("Predict"):
                predictions=model.predict([[year,area,season,cropno]])
                st.write("The predicted crop production is: ", predictions)

    elif selectModel == "Support Vector Machiene":


        st.subheader("Support Vector Machine" )
        selectKernel = st.sidebar.selectbox("Select Kernel", options=["Select Kernel","Poly","RBF","Linear","Sigmoid"])

        if selectKernel == "Poly" :

            svrpoly = SVR(kernel = 'poly')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("Poly Kernel")
                svrpoly.fit(x_train, y_train)
                predictionpoly = svrpoly.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionpoly)
                msepoly= mean_squared_error(y_test,predictionpoly)
                st.write("Mean Squared Error: ",msepoly)
                with open("SVM.txt", "a") as file:
                    file.write("PolyP ")
                    file.write(str(msepoly) +'\n')

            if manual_predict:
                svrpoly.fit(data_input.values,data_target)
                joblib.dump(svrpoly,'crop-productionpoly.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                svrpoly=joblib.load('crop-productionpoly.joblib')
                if st.button("Predict"):
                    predictions=svrpoly.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictions)



        elif selectKernel == "Linear" :

            svrlinear = SVR(kernel = 'linear')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("Linear Kernel")
                svrlinear.fit(x_train, y_train)
                predictionlinear = svrlinear.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionlinear)
                mselinear = mean_squared_error(y_test,predictionlinear)
                st.write("Mean Squared Error: ",mselinear)
                with open("SVM.txt", "a") as file:
                    file.write("LinearP ")
                    file.write(str(mselinear) +'\n')

            if manual_predict:

                svrlinear.fit(data_input.values,data_target)
                joblib.dump(svrlinear,'crop-productionlinear.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                svrlinear=joblib.load('crop-productionlinear.joblib')

                if st.button("Predict"):
                    predictions=svrlinear.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictions)



        elif selectKernel == "RBF" :

            svrrbf = SVR(kernel = 'rbf')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("RBF Kernel")
                svrrbf.fit(x_train, y_train)
                predictionrbf = svrrbf.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionrbf)
                mserbf = mean_squared_error(y_test,predictionrbf)
                st.write("Mean Squared Error: ",mserbf)
                with open("SVM.txt", "a") as file:
                    file.write("RbfP ")
                    file.write(str(mserbf) +'\n')

            if manual_predict:

                svrrbf.fit(data_input.values,data_target)
                joblib.dump(svrrbf,'crop-productionrbf.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")

                svrrbf=joblib.load('crop-productionrbf.joblib')

                if st.button("Predict"):
                    predictions=svrrbf.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is", predictions)


        elif selectKernel == "Sigmoid" :

             svrsigmoid = SVR(kernel = 'sigmoid')
             auto_input_testing=st.checkbox("Auto Input Testing")
             manual_predict = st.checkbox("User Input Testing")

             if auto_input_testing:

                st.subheader("Sigmoid Kernel")
                svrsigmoid.fit(x_train, y_train)
                predictionsigmoid = svrsigmoid.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionsigmoid)
                msesigmoid = mean_squared_error(y_test,predictionsigmoid)
                st.write("Mean Squared Error: ",msesigmoid)
                with open("SVM.txt", "a") as file:
                    file.write("SigmoidP ")
                    file.write(str(msesigmoid) +'\n')

             if manual_predict:

                svrsigmoid.fit(data_input.values,data_target)
                joblib.dump(svrsigmoid,'crop-productionsigmoid.joblib')
                year = st.number_input("Enter year:")
                area = st.number_input("Enter area:")
                season = st.number_input("Enter season:")
                cropno= st.number_input("Enter crop number:")
                random= st.number_input("Enter random:")


                svrsigmoid=joblib.load('crop-productionsigmoid.joblib')
                if st.button("Predict"):
                    predictions=svrsigmoid.predict([[year,area,season,cropno]])
                    st.write("The predicted crop production is: ", predictions)
    showConclusion = st.sidebar.checkbox("Show Conclusion")
    if showConclusion:
        st.empty()
        df = pd.read_csv("SVM.txt", delim_whitespace=True, names=["type", "acc"])

        polyAccuraccyW = df.loc[df['type'] == 'PolyP','acc'].values[0]
        rbfAccuraccyW = df.loc[df['type'] == 'RbfP','acc'].values[0]
        sigmoidAccuraccyW = df.loc[df['type'] == 'SigmoidP','acc'].values[0]
        linearAccuraccyW = df.loc[df['type'] == 'LinearP','acc'].values[0]
        dataSVMW = [
        ['Kernel Name', 'MSE Value'],
        ['Poly', polyAccuraccyW],
        ['RBF', rbfAccuraccyW],
        ['Linear', linearAccuraccyW],
        ['Sigmoid', sigmoidAccuraccyW]
        ]
        st.header("Summary For SVM")
        st.table(dataSVMW)
        maxSvmW = polyAccuraccyW;
        maxSvmKernelW = "Poly"
        if(maxSvmW > rbfAccuraccyW):
            maxSvmW = rbfAccuraccyW
            maxSvmKernelW = "RBF"
        if(maxSvmW > linearAccuraccyW):
            maxSvmW = linearAccuraccyW
            maxSvmKernelW = "Linear"
        if(maxSvmW > sigmoidAccuraccyW):
            maxSvmW = sigmoidAccuraccyW
            maxSvmKernelW = "Sigmoid"
        st.write("The most  accurate MSE value between all kernel is kernel ", maxSvmKernelW," with MSE value is " , maxSvmW)

        df = pd.read_csv("KNN.txt", delim_whitespace=True, names=["number", "acc"])
        aW = df.loc[df['number'] == '150NP','acc'].values[0]
        bW = df.loc[df['number'] == '100NP','acc'].values[0]
        cW = df.loc[df['number'] == '50NP','acc'].values[0]
        dW = df.loc[df['number'] == '1NP','acc'].values[0]
        dataKNNW = [
        ['Number of Neighbour', 'MSE Value'],
        ['150', aW],
        ['100', bW],
        ['50', cW],
        ['1', dW]
        ]
        st.header("Summary For KNN")
        st.table(dataKNNW)
        maxKnnW = aW
        maxKnnNumberW = "150"

        if(maxKnnW >bW):
            maxKnnW = bW
            maxKnnNumberW = "100"

        if(maxKnnW > cW):
            maxKnnW = cW
            maxKnnNumberW = "50"

        if(maxKnnW > dW):
            maxKnnW = dW
            maxKnnNumberW = "1"

        st.write("The most highest accuraccy between all neighbour is ", maxKnnNumberW," neighbors with MSE value is " , maxKnnW)

        df = pd.read_csv("DT.txt", delim_whitespace=True, names=["name", "acc"])
        maxDecisionTreeW = df.loc[df['name'] == 'DTP','acc'].values[0]
        dataDTW = [
        ['Model Name', 'MSE Value'],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.header("Summary For Decision Tree")
        st.table(dataDTW)
        st.write("The MSE value for Decision Tree model is " , maxDecisionTreeW)


        st.header("Summary For All 3 Model")
        st.subheader("Below is the most accurate MSE Value for each model")
        headSVMW = "SVM kernel = "
        headKNNW = "KNN neighbors = "
        headSVMW += maxSvmKernelW
        headKNNW += maxKnnNumberW
        dataAllW = [
        ['Model Name', 'MSE Value'],
        [headSVMW, maxSvmW],
        [headKNNW, maxKnnW],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.table(dataAllW)
        maxAllW = maxSvmW
        maxAllStringW = headSVMW
        if(maxAllW > maxKnnW):
            maxAllW = maxKnnW
            maxAllStringW = headKNNW
        if(maxAllW > maxDecisionTreeW):
            maxAllW = maxDecisionTreeW
            maxAllStringW = "Decision Tree"
        st.write("The most acurate between the most acurate for each 3 model is ", maxAllStringW," with MSE value is ", maxAllW)
elif(selectDataset == "Crop Climate"):#imran
    import pandas as pd
    import numpy as np
    import streamlit as st
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    import joblib
    st.title("Crop Climate")
    st.write("Climate refers to the long-term patterns of temperature, precipitation, wind, and other meteorological factors that occur in a particular region. These patterns can have a significant impact on crop growth and productivity. Crop growth is optimal within a certain temperature range and when there is enough sunlight and water. For example, warm-season crops such as corn and soybeans require warm temperatures and a long growing season, while cool-season crops such as wheat and barley can tolerate cooler temperatures and shorter growing seasons. Precipitation also plays a crucial role in crop growth, as it provides the water needed for plants to grow and develop. In regions with high rainfall, crops such as rice and sugarcane can thrive, while in regions with low rainfall, drought-resistant crops such as sorghum and millet may be more appropriate. Climate change is expected to have a significant impact on crop growth and productivity in the future. Rising temperatures and changing precipitation patterns are likely to affect the suitability of different regions for different crops, and may also lead to increased pest and disease pressure. Adaptation strategies such as crop diversification, irrigation, and the use of drought-resistant varieties will be important for maintaining crop productivity in a changing climate.")
    data_cp = pd.read_csv("climate-ds.csv")
    data_input = data_cp.drop(columns = ['Id', 'Area', 'Item', 'hg/ha_yield'])
    data_target = data_cp["hg/ha_yield"]
    st.write("sini")
    data_input
    x_train,x_test,y_train,y_test = train_test_split(data_input,data_target,test_size = 0.20)

    selectModel = st.sidebar.selectbox ("Select Model", options = ["Select Model","Support Vector Machiene","K-Nearest Neighbors","Decission Tree"])

    if selectModel == "K-Nearest Neighbors":
        st.subheader("K-Nearest Neighbor" )
        selectDistance = st.sidebar.selectbox("Select Distance", options=["Select Distance","150","100","50","1"])

        if selectDistance == "150":

            kn150=KNeighborsRegressor (n_neighbors = 150)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=150")
                kn150.fit(x_train,y_train)
                prediction1knn150=kn150.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=150 Data Target",prediction1knn150)
                mseknn150 = mean_squared_error(y_test,prediction1knn150)
                st.write("Mean Squared Error: ",mseknn150)
                with open("KNN.txt", "a") as file:
                    file.write("150NY ")
                    file.write(str(mseknn150) +'\n')

            if manual_predict:
                kn150.fit(data_input.values,data_target)
                joblib.dump(kn150, 'crop-climateknn150.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                model=joblib.load('crop-climateknn150.joblib')

                if st.button("Predict"):
                    predictionkn150=kn150.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictionkn150)

        elif selectDistance == "100":

            kn100=KNeighborsRegressor (n_neighbors = 100)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=100")
                kn100.fit(x_train,y_train)
                prediction1knn100=kn100.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=100 Data Target",prediction1knn100)
                mseknn100 = mean_squared_error(y_test,prediction1knn100)
                st.write("Mean Squared Error: ",mseknn100)
                with open("KNN.txt", "a") as file:
                    file.write("100NY ")
                    file.write(str(mseknn100) +'\n')

            if manual_predict:

                kn100.fit(data_input.values,data_target)
                joblib.dump(kn100, 'crop-climateknn100.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                model=joblib.load('crop-climateknn100.joblib')

                if st.button("Predict"):
                    predictionkn100=kn100.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictionkn100)


        elif selectDistance == "50":

            kn50=KNeighborsRegressor (n_neighbors = 50)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=50")
                kn50.fit(x_train,y_train)
                predictionknn50=kn50.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=50 Data Target",predictionknn50)
                mseknn50 = mean_squared_error(y_test,predictionknn50)
                st.write("Mean Squared Error: ",mseknn50)
                with open("KNN.txt", "a") as file:
                    file.write("50NY ")
                    file.write(str(mseknn50) +'\n')

            if manual_predict:

                kn50.fit(data_input.values,data_target)
                joblib.dump(kn50, 'crop-climateknn50.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                model=joblib.load('crop-climateknn50.joblib')

                if st.button("Predict"):
                    predictionkn50=kn50.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictionkn50)


        elif selectDistance == "1":

            kn1=KNeighborsRegressor (n_neighbors = 1)
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:
                st.subheader("n_neighbor=1")
                kn1.fit(x_train,y_train)
                prediction1knn1=kn1.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("n_neighbor=1 Data Target",prediction1knn1)
                mseknn1 = mean_squared_error(y_test,prediction1knn1)
                st.write("Mean Squared Error: ",mseknn1)
                with open("KNN.txt", "a") as file:
                    file.write("1NY ")
                    file.write(str(mseknn1) +'\n')

            if manual_predict:

                kn1.fit(data_input.values,data_target)
                joblib.dump(kn1, 'crop-climateknn1.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                model=joblib.load('crop-climateknn1.joblib')

                if st.button("Predict"):
                    predictionknn1=kn1.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictionknn1)


    elif selectModel == "Decission Tree":

        model = DecisionTreeRegressor()
        auto_input_testing=st.checkbox("Auto Input Testing")
        manual_predict = st.checkbox("User Input Testing")

        if auto_input_testing:
            model.fit(x_train,y_train)
            predictiondt=model.predict(x_test)
            st.write("Testing Data Target",y_test)
            st.write("Decision Tree Data Target",predictiondt)
            score = mean_squared_error(y_test,predictiondt)
            st.write("Mean Squared Error: ",score)
            with open("DT.txt", "a") as file:
                file.write("DTY ")
                file.write(str(score) +'\n')

        if manual_predict:
            model.fit(data_input.values,data_target)
            joblib.dump(model, 'crop-production.joblib')
            year = st.number_input("Enter year:")
            average = st.number_input("Enter average rainfall per year:")
            pesticide = st.number_input("Enter pesticide tonnes:")
            temperature= st.number_input("Enter temperature average:")
            item= st.number_input("Enter item:")

            model=joblib.load('crop-production.joblib')

            if st.button("Predict"):
                predictions=model.predict([[year,average,pesticide, temperature ,item]])
                st.write("The predicted crop yield is: ", predictions)

    elif selectModel == "Support Vector Machiene":


        st.subheader("Support Vector Machine" )
        selectKernel = st.sidebar.selectbox("Select Kernel", options=["Select Kernel","Poly","RBF","Linear","Sigmoid"])

        if selectKernel == "Poly" :

            svrpoly = SVR(kernel = 'poly')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("Poly Kernel")
                svrpoly.fit(x_train, y_train)
                predictionpoly = svrpoly.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionpoly)
                msepoly= mean_squared_error(y_test,predictionpoly)
                st.write("Mean Squared Error: ",msepoly)
                with open("SVM.txt", "a") as file:
                    file.write("PolyY ")
                    file.write(str(msepoly) +'\n')

            if manual_predict:
                svrpoly.fit(data_input.values,data_target)
                joblib.dump(svrpoly,'crop-productionpoly.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                svrpoly=joblib.load('crop-productionpoly.joblib')
                if st.button("Predict"):
                    predictions=svrpoly.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictions)



        elif selectKernel == "Linear" :

            svrlinear = SVR(kernel = 'linear')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("Linear Kernel")
                svrlinear.fit(x_train, y_train)
                predictionlinear = svrlinear.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionlinear)
                mselinear = mean_squared_error(y_test,predictionlinear)
                st.write("Mean Squared Error: ",mselinear)
                with open("SVM.txt", "a") as file:
                    file.write("LinearY ")
                    file.write(str(mselinear) +'\n')

            if manual_predict:

                svrlinear.fit(data_input.values,data_target)
                joblib.dump(svrlinear,'crop-productionlinear.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                svrlinear=joblib.load('crop-productionlinear.joblib')

                if st.button("Predict"):
                    predictions=svrlinear.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictions)



        elif selectKernel == "RBF" :

            svrrbf = SVR(kernel = 'rbf')
            auto_input_testing=st.checkbox("Auto Input Testing")
            manual_predict = st.checkbox("User Input Testing")

            if auto_input_testing:

                st.subheader("RBF Kernel")
                svrrbf.fit(x_train, y_train)
                predictionrbf = svrrbf.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionrbf)
                mserbf = mean_squared_error(y_test,predictionrbf)
                st.write("Mean Squared Error: ",mserbf)
                with open("SVM.txt", "a") as file:
                    file.write("RbfY ")
                    file.write(str(mserbf) +'\n')

            if manual_predict:

                svrrbf.fit(data_input.values,data_target)
                joblib.dump(svrrbf,'crop-productionrbf.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")

                svrrbf=joblib.load('crop-productionrbf.joblib')

                if st.button("Predict"):
                    predictions=svrrbf.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is", predictions)


        elif selectKernel == "Sigmoid" :

             svrsigmoid = SVR(kernel = 'sigmoid')
             auto_input_testing=st.checkbox("Auto Input Testing")
             manual_predict = st.checkbox("User Input Testing")

             if auto_input_testing:

                st.subheader("Sigmoid Kernel")
                svrsigmoid.fit(x_train, y_train)
                predictionsigmoid = svrsigmoid.predict(x_test)
                st.write("Testing Data Target",y_test)
                st.write("Decision Tree Data Target",predictionsigmoid)
                msesigmoid = mean_squared_error(y_test,predictionsigmoid)
                st.write("Mean Squared Error: ",msesigmoid)
                with open("SVM.txt", "a") as file:
                    file.write("SigmoidY ")
                    file.write(str(msesigmoid) +'\n')

             if manual_predict:

                svrsigmoid.fit(data_input.values,data_target)
                joblib.dump(svrsigmoid,'crop-productionsigmoid.joblib')
                year = st.number_input("Enter year:")
                average = st.number_input("Enter average rainfall per year:")
                pesticide = st.number_input("Enter pesticide tonnes:")
                temperature= st.number_input("Enter temperature average:")
                item= st.number_input("Enter item:")


                svrsigmoid=joblib.load('crop-productionsigmoid.joblib')
                if st.button("Predict"):
                    predictions=svrsigmoid.predict([[year,average,pesticide, temperature ,item]])
                    st.write("The predicted crop yield is: ", predictions)
    showConclusion = st.sidebar.checkbox("Show Conclusion")
    if showConclusion:
        st.empty()
        df = pd.read_csv("SVM.txt", delim_whitespace=True, names=["type", "acc"])

        polyAccuraccyW = df.loc[df['type'] == 'PolyP','acc'].values[0]
        rbfAccuraccyW = df.loc[df['type'] == 'RbfP','acc'].values[0]
        sigmoidAccuraccyW = df.loc[df['type'] == 'SigmoidP','acc'].values[0]
        linearAccuraccyW = df.loc[df['type'] == 'LinearP','acc'].values[0]
        dataSVMW = [
        ['Kernel Name', 'MSE Value'],
        ['Poly', polyAccuraccyW],
        ['RBF', rbfAccuraccyW],
        ['Linear', linearAccuraccyW],
        ['Sigmoid', sigmoidAccuraccyW]
        ]
        st.header("Summary For SVM")
        st.table(dataSVMW)
        maxSvmW = polyAccuraccyW;
        maxSvmKernelW = "Poly"
        if(maxSvmW > rbfAccuraccyW):
            maxSvmW = rbfAccuraccyW
            maxSvmKernelW = "RBF"
        if(maxSvmW > linearAccuraccyW):
            maxSvmW = linearAccuraccyW
            maxSvmKernelW = "Linear"
        if(maxSvmW > sigmoidAccuraccyW):
            maxSvmW = sigmoidAccuraccyW
            maxSvmKernelW = "Sigmoid"
        st.write("The most  accurate MSE value between all kernel is kernel ", maxSvmKernelW," with MSE value is " , maxSvmW)

        df = pd.read_csv("KNN.txt", delim_whitespace=True, names=["number", "acc"])
        aW = df.loc[df['number'] == '150NP','acc'].values[0]
        bW = df.loc[df['number'] == '100NP','acc'].values[0]
        cW = df.loc[df['number'] == '50NP','acc'].values[0]
        dW = df.loc[df['number'] == '1NP','acc'].values[0]
        dataKNNW = [
        ['Number of Neighbour', 'MSE Value'],
        ['150', aW],
        ['100', bW],
        ['50', cW],
        ['1', dW]
        ]
        st.header("Summary For KNN")
        st.table(dataKNNW)
        maxKnnW = aW
        maxKnnNumberW = "150"

        if(maxKnnW >bW):
            maxKnnW = bW
            maxKnnNumberW = "100"

        if(maxKnnW > cW):
            maxKnnW = cW
            maxKnnNumberW = "50"

        if(maxKnnW > dW):
            maxKnnW = dW
            maxKnnNumberW = "1"

        st.write("The most highest accuraccy between all neighbour is ", maxKnnNumberW," neighbors with MSE value is " , maxKnnW)

        df = pd.read_csv("DT.txt", delim_whitespace=True, names=["name", "acc"])
        maxDecisionTreeW = df.loc[df['name'] == 'DTP','acc'].values[0]
        dataDTW = [
        ['Model Name', 'MSE Value'],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.header("Summary For Decision Tree")
        st.table(dataDTW)
        st.write("The MSE value for Decision Tree model is " , maxDecisionTreeW)


        st.header("Summary For All 3 Model")
        st.subheader("Below is the most accurate MSE Value for each model")
        headSVMW = "SVM kernel = "
        headKNNW = "KNN neighbors = "
        headSVMW += maxSvmKernelW
        headKNNW += maxKnnNumberW
        dataAllW = [
        ['Model Name', 'MSE Value'],
        [headSVMW, maxSvmW],
        [headKNNW, maxKnnW],
        ['Decision Tree', maxDecisionTreeW]
        ]
        st.table(dataAllW)
        maxAllW = maxSvmW
        maxAllStringW = headSVMW
        if(maxAllW > maxKnnW):
            maxAllW = maxKnnW
            maxAllStringW = headKNNW
        if(maxAllW > maxDecisionTreeW):
            maxAllW = maxDecisionTreeW
            maxAllStringW = "Decision Tree"
        st.write("The most acurate between the most acurate for each 3 model is ", maxAllStringW," with MSE value is ", maxAllW)
