import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import math
import nltk
from sklearn.model_selection import train_test_split
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



def data_processing(tdata,algorithm):
     
    classifier = None
    evaluations = None

    # Training the model using Multinomial Naive Bayes classifier
    def Navies_Bayesian_Model(training_data,y_train):
        classifier = MultinomialNB()
        classifier.fit(training_data, y_train)
        return classifier

    # Training the model using SVM classifier    
    def SVM_Model(training_data,y_train):
        classifier = svm.SVC(kernel='rbf') # rbf Kernel
        classifier.fit(training_data, y_train)
        return classifier

    # Training the model using Decision Tree classifier
    def Decision_Tree_Model(training_data,y_train):
        classifier = DecisionTreeClassifier()
        classifier.fit(training_data, y_train)
        return classifier

    # Training the model using Logistic Regression classifier
    def Logistic_Regression_Model(training_data,y_train):
        classifier = LogisticRegression()
        classifier.fit(training_data, y_train)
        return classifier

    # Training the model using KNN classifier
    def KNN_Model(training_data,y_train):
        classifier = KNeighborsClassifier()
        classifier.fit(training_data, y_train)
        return classifier

    # Function for Evaluating the prediction
    def getpredictionEvaluations(y_test,predictions):
        print(" Start getpredictionEvaluations ")
        res = {"Accuracy score: ": accuracy_score(y_test, predictions),
                "Recall score: ": recall_score(y_test, predictions, average = 'weighted'),
                "Precision score: ": precision_score(y_test, predictions, average = 'weighted'),
                "F1 score: ": f1_score(y_test, predictions, average = 'weighted')}
        print("Accuracy score: ", accuracy_score(y_test, predictions))
        print("Recall score: ", recall_score(y_test, predictions, average = 'weighted'))
        print("Precision score: ", precision_score(y_test, predictions, average = 'weighted'))
        print("F1 score: ", f1_score(y_test, predictions, average = 'weighted'))
        print(" End getpredictionEvaluations ")
        return res

    # Function to display Evaluations
    def show_evaluations(evaluations):
        st.write(evaluations)

    #Cleaning of the data 
    tdata.dropna(axis=0,inplace=True)
    print(tdata.shape)

    tdata.drop(['urgency','impact','ticket_type'],axis=1,inplace=True)
    print(tdata.shape)

    #Spliting data into training and testing data
    Y = tdata['category']
    X = tdata.drop(columns=['title','category'])

    X_train1, X_test1, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


    X_train=X_train1['body']
    X_test=X_test1['body']


    print("Training dataset: ", X_train.shape[0])
    print("Test dataset: ", X_test.shape[0])

    #Applying bag of words processing to the dataset
    count_vector = CountVectorizer(stop_words = 'english')
    training_data = count_vector.fit_transform(X_train)
    testing_data = count_vector.transform(X_test)

    
    tfidf_transformer = TfidfTransformer()
    training_data= tfidf_transformer.fit_transform(training_data)
    print(training_data.shape)
    print(algorithm)


    if algorithm == "Navies Bayesian":
        classifier = Navies_Bayesian_Model(training_data,y_train,)
    elif algorithm == "Decision Tree/Random Forest":
        classifier = Decision_Tree_Model(training_data,y_train)
    elif algorithm == "Support Vector Machine":
        classifier = SVM_Model(training_data,y_train)
    elif algorithm == "K-nearest Neigbhour":
        classifier = KNN_Model(training_data,y_train)
    elif algorithm == "Logistic Regression":
        classifier = Logistic_Regression_Model(training_data,y_train)
    
    # Generating predictions

    predictions = classifier.predict(testing_data)
    print(predictions)   
    evaluations = getpredictionEvaluations(y_test,predictions)
    show_evaluations(evaluations)

    X_test1['category'] = predictions
    
    def show_bar_plot(df,cname):
        df.groupby(cname).body.count().plot.bar(ylim=0)
        # plt.show()
        st.pyplot(plt)
        sns.catplot(x="category", y="No of tickets", hue="sub_category1", kind="bar", data=df) 
        # chart_data = pd.DataFrame(df)
        # st.bar_chart(df,columns=[cname])
    show_bar_plot(X_test1,"category")



#Function to upload datset into dataFrame
def dataset_upload(alg):
    Uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if st.button("Process"):
        if Uploaded_file is not None:
            file_details = {"Filename":Uploaded_file.name,"FileType":Uploaded_file.type,"FileSize":Uploaded_file.size}
            st.write(file_details)

            df = pd.read_csv(Uploaded_file)
            st.dataframe(df.head())
            data_processing(df,alg)
             


def app_main():
    print(" main start ")
    st.title("Support Ticket Classifier")

    algos = ["Navies Bayesian","Decision Tree/Random Forest", "Support Vector Machine", "K-nearest Neigbhour", "Logistic Regression"]
    algorithm = st.sidebar.selectbox("Select Algorithm", algos)

    if algorithm == "Navies Bayesian":
        st.subheader("Navies Bayesian Algorithm")

    elif algorithm == "Decision Tree/Random Forest":
        st.subheader("Decision Tree/Random Forest Algorithm")

    elif algorithm == "Support Vector Machine":
        st.subheader("Support Vector Machine Algorithm")

    elif algorithm == "K-nearest Neigbhour":
        st.subheader("K-nearest Neigbhour Algorithm")

    else:
        st.subheader("Logistic Regression Algorithm")

    dataset_upload(algorithm)
    print(" main end ")

app_main()
