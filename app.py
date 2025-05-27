import streamlit as st




from datetime import datetime
import pandas as pd


import plotly.express as px
import plotly.graph_objects as gopip

import pickle as pkle
import os.path


from copyreg import pickle
import math
from math import sqrt
import pickle
import plotly.express as px
from sklearn.tree import _tree
import re

from sklearn.inspection import PartialDependenceDisplay
from streamlit_echarts import st_echarts
from streamlit_plotly_events import plotly_events
import json


from IPython.display import display
import pandas as pd

#
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


st.set_page_config(layout="wide")

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt



import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# from witwidget.notebook.visualization import WitWidget, WitConfigBuilder


from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns




from sklearn.metrics import accuracy_score
import pandas as pd
import shap


st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_option("deprecation.showfileUploaderEncoding", False)


hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}
</style>

"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# st.title('XAI Methods On Human-AI Cooperation Spectrum')

# st.title("""
#  Visually Explore Machine Learning Prediction
# """)


from io import BytesIO, StringIO
from typing import Union

import pandas as pd
import streamlit as st
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')



# FILE_TYPES = ["csv",'xlsx']

hide_streamlit_style = """
<style>
.css-hi6a2p {padding-top: 0rem;}
</style>
"""


st.write(
    """
**Task Description**

You are a loan approval officer and you are verifying recommendations made by a Machine learning model.
You have the access to the explanations of AI decisions. You can adopt or override AI decisions.
Your job is to make as accurate decisions as possible which means reject applicants with high risks and approve applicants with low risks.

"""
)

st.write("\n")

def load_process_data(filename):
    data = pd.read_csv(filename, engine="python", index_col=False)
    # remove first column
    data = data.iloc[:, 1:]

    ### remove special values

    cols_name = data.columns
    for cols in cols_name:
        data = data[data[cols] != -9]
        data = data[data[cols] != -8]
        data = data[data[cols] != -7]

    return data

df = load_process_data('heloc_dataset_use.csv')

 
def split_df(data):

    variable_names = list(data.columns[1:])

    X = data[variable_names]

    data["RiskPerformance"] = np.where(data["RiskPerformance"] == "Bad", 1, 0)

    y = data["RiskPerformance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=83
    )

    return X, y, X_train, X_test, y_train, y_test


X, y, X_train, X_test, y_train, y_test = split_df(df)


variable_names = list(df.columns[1:])

testid = [i for i in range(1, len(X_test))]

with st.container():

    st.write("""
**Check New Applications**
""" )
    Newapp = st.selectbox(
        'Choose the application id in the test set',testid ,key=str(1) )

    st.write("The information for your selected application is shown in the below table" )
    st.dataframe(X_test.iloc[[Newapp]])

# with st.expander("Diagnose New Patients"):
#     NewPatients = st.selectbox(
#         'Choose the patient id in the test set',testid ,key=str(1) )
#     st.dataframe(X_test.iloc[[NewPatients-1]])

#     Diagnose = st.selectbox(
#         'Choose your diagnosis: The tumor of this patient is ', ('', 'malignant', 'benign'))

#     rightanswer = y_test.iloc[NewPatients - 1]


class Model:
    def __init__(self):

        self.model = pickle.load(
            open(
                "decision_tree.pkl","rb"))

    def test_model(self):
        if not self.model:
            print("Train Model First")

        self.train_pred = self.model.predict(X_train)
        self.test_pred = self.model.predict(X_test)

        self.y_train_df = pd.DataFrame(y_train)
        # self.train_pred_df = pd.DataFrame(self.train_pred, columns= ['Predict'])
        self.y_train_df["Predict"] = self.train_pred

        self.y_test_df = pd.DataFrame(y_test)
        # self.train_pred_df = pd.DataFrame(self.train_pred, columns= ['Predict'])
        self.y_test_df["Predict"] = self.test_pred

        acc_train = round((np.mean(self.train_pred == y_train) * 100), 2)
        acc_test = round((np.mean(self.test_pred == y_test) * 100), 2)

        self.train = pd.concat([X_train, self.y_train_df], axis=1)
        self.test = pd.concat([X_test, self.y_test_df], axis=1)

        # self.test.to_csv("/Users/ypi/Desktop/xai_data/heloc_dataset_test.csv")

        # st.write("Training Accuracy:", acc_train, '%')
        # st.write("Test Accuracy:", acc_test, '%')

    def shapfeatureimportance(self, idnum):
        self.explainer = shap.TreeExplainer(self.model)
        data_for_prediction = X_test.iloc[[idnum]]
        self.features = X_test.columns
    
        # shap_values is a 3D numpy array: (1, num_features, num_classes)
        self.shap_values = self.explainer.shap_values(data_for_prediction)
    
        predicted_class = self.test.iloc[idnum]["Predict"]
    
        if predicted_class == 1:
            # Get SHAP values for class 1 for this single instance
            feature_importances = self.shap_values[0, :, 1]
    
            # Features with positive importance for class 1
            indexes = [i for i, val in enumerate(feature_importances) if val > 0]
    
            self.shapdf = pd.DataFrame({
                "Feature": self.features,
                "Feature Importance": feature_importances,
                "Feature Value": data_for_prediction.values[0]
            })
    
            self.makeitfeature = [self.features[i] for i in indexes]
    
        else:
            # Get SHAP values for class 0
            feature_importances = self.shap_values[0, :, 0]
    
            # Features with positive importance for class 0
            indexes = [i for i, val in enumerate(feature_importances) if val > 0]
    
            self.shapdf = pd.DataFrame({
                "Feature": self.features,
                "Feature Importance": feature_importances,
                "Feature Value": data_for_prediction.values[0]
            })
    
            self.makeitfeature = [self.features[i] for i in indexes]
    
            # Visualize force plot for class 1 (optional)
            shap.initjs()
            display(
                shap.force_plot(
                    self.explainer.expected_value[1],
                    self.shap_values[0, :, 1],
                    data_for_prediction,
                )
            )
    
        return self.makeitfeature, self.shapdf


    def importfeaturedict(self, idnum):
        self.featuredict = dict()

        for i in self.makeitfeature:
            featurevalue = self.test[i].iloc[idnum]
            self.featuredict[i] = featurevalue

        return self.featuredict

    def featurebar(self, idnum):

        


        self.shapdf["Color"] = np.where(self.shapdf["Feature Importance"]>0, 'red', 'blue')


        fig = px.bar(self.shapdf, x="Feature Importance", y="Feature", orientation="h",text= "Feature Value")
        
        
        
        

        # fig=go.Figure(data=go.Bar(x=self.shapdf['Feature Importance'], y=self.shapdf['Feature'],orientation='h'))

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"}, hovermode="y", height=500)
        
        fig.update_layout(height=500,width=650)

                   
        fig.update_traces(marker_color=self.shapdf["Color"])

        self.selectedfeature = plotly_events(fig)


        st.write("The chart above shows for each feature of the selected application, whether it increases(red bars) or decreases(blue bars) applicants' chance of repaying, and by how much.")
        st.write("""
        **Try to click a bar to see overall model performance.**
        """)




        self.click  = False

        try:

            self.select = self.selectedfeature[0]["y"]

            self.click  = True 
            return self.select

        except IndexError:
            pass
            
            
        

            

        # fig["data"][0]["marker"]["color"] = [
        #     "red" if c == self.select else "blue" for c in fig["data"][0]["y"]
        # ]

        # st.write(fig["data"][0]["marker"]["color"])

        

    def showconfusion(self, y_true, predict, class_name):
        newcol = []
        y_true = y_true.tolist()
        predict = predict.tolist()
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == class_name:
                if y_true[i] == predict[i]:
                    TP += 1
                    newcol.append("High Risk-Predicted Correctly")

                else:
                    FN += 1
                    newcol.append("High Risk-Predicted Incorrectly")
            else:
                if y_true[i] == predict[i]:
                    TN += 1
                    newcol.append("Low Risk-Predicted Correctly")
                else:
                    FP += 1
                    newcol.append("Low Risk-Predicted Incorrectly")
        martix = np.array([[TP, FP], [FN, TN]])

        plt.figure(figsize=(1, 1))
        sns.set(font_scale=1.8)

        ax = sns.heatmap(
            martix,
            xticklabels="10",
            yticklabels="10",
            annot=True,
            square=True,
            cmap="Blues",
            fmt=".4g",
            annot_kws={"size": 4},
            cbar=False,
        )

        ax.set_xticklabels("10", size=4)
        ax.set_yticklabels("10", size=4)
        ax.set_xlabel("Actual", fontsize=4)
        ax.set_ylabel("Predicted", fontsize=4)
        # st.pyplot()

    def showmodelperf(self, y_true, predict, class_name):
        newcol = []
        y_true = y_true.tolist()
        predict = predict.tolist()
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(y_true)):
            if y_true[i] == class_name:
                if y_true[i] == predict[i]:
                    TP += 1
                    newcol.append("High Risk-Predicted Correctly")

                else:
                    FN += 1
                    newcol.append("High Risk-Predicted Incorrectly")
            else:
                if y_true[i] == predict[i]:
                    TN += 1
                    newcol.append("Low Risk-Predicted Correctly")
                else:
                    FP += 1
                    newcol.append("Low Risk-Predicted Incorrectly")

        self.train["Model Performance"] = newcol


    def verbalexp(self, idnum):

        predresult = "low" if self.test.iloc[idnum]["Predict"] == 0 else "high"
        predverb = (
            "The model predicted that the applicant have %s risk of repaying loans "
            % (predresult)
        )
        featurename = self.select
        featurevalue = self.test.iloc[idnum][featurename]
        subdf = self.train[(self.train[featurename] == featurevalue)]
        lensubdf = len(subdf)

        if predresult == "low":
            corr = len(
                subdf[
                    subdf["Model Performance"].str.contains(
                        "Low Risk-Predicted Correctly"
                    )
                ]
            )
            incorr = len(
                subdf[
                    subdf["Model Performance"].str.contains(
                        "Low Risk-Predicted Incorrectly"
                    )
                ]
            )
            total = corr + incorr
            corracc = (corr / total) * 100
        else:
            corr = len(
                subdf[
                    subdf["Model Performance"].str.contains(
                        "High Risk-Predicted Correctly"
                    )
                ]
            )
            incorr = len(
                subdf[
                    subdf["Model Performance"].str.contains(
                        "High Risk-Predicted Incorrectly"
                    )
                ]
            )
            total = corr + incorr
            corracc = (corr / total) * 100

        expfeature = (
            "In the training set, %d applications share the same value for %s as the applicant you are investigating. The model predcits %d of them as %s risk. "
            % (lensubdf, featurename, total, predresult)
        )
        expacc = (
            "Among those %d applications predicted as %s risk, %d are predicted correctly, %d are predicted incorrectly. %d percent of applications are predicted correctly %s risk"
            % (total, predresult, corr, incorr, corracc,predresult)
        )

        st.write(predverb)
        st.write(expfeature + expacc)

    def plotdisforimportantfeaturewithperf2(self, idnum):

        category_orders = [
            "High Risk-Predicted Incorrectly",
            "High Risk-Predicted Correctly",
            "Low Risk-Predicted Incorrectly",
            "Low Risk-Predicted Correctly",
        ]

        colors_dict = {
            "High Risk-Predicted Incorrectly": "rgb(161,40,48)",
            "High Risk-Predicted Correctly": "rgb(240,17,0)",
            "Low Risk-Predicted Incorrectly": "rgb(143,151,121)",
            "Low Risk-Predicted Correctly": "rgb(79,121,66)",
        }
        
        plotly_colors = [colors_dict[c] for c in category_orders]

        fig1 = px.histogram(
            self.train,
            x=list(self.featuredict)[0],
            color="Model Performance",
            category_orders={
                "Model Performance": [
                    "High Risk-Predicted Incorrectly",
                    "High Risk-Predicted Correctly",
                    "Low Risk-Predicted Incorrectly",
                    "Low Risk-Predicted Correctly",
                ]
            },
            color_discrete_sequence=plotly_colors,
        )

        fig1.add_annotation(
            text="Your datapoint is here",
            showarrow=True,
            arrowhead=1,
            x=list(self.featuredict.values())[0],
            y=0.2,
            align="left",
            xanchor="left",
            font=dict(size=15, color="#242526"),
        )

        fig1.update_traces(marker_line_width=2)

        st.plotly_chart(fig1, use_container_width=True)

        selected_points = plotly_events(fig1)

        self.verbalexp(idnum)

        if selected_points:

            selectdf = self.train[(self.train[list(self.featuredict)[0]] == xvalue)]

            # if categoryvalue==0:

            fig2 = px.histogram(
                selectdf,
                x=list(self.featuredict)[1],
                color="Model Performance",
                category_orders={
                    "Model Performance": [
                        "High Risk-Predicted Incorrectly",
                        "High Risk-Predicted Correctly",
                        "Low Risk-Predicted Incorrectly",
                        "Low Risk-Predicted Correctly",
                    ]
                },
            )

            fig2.update_traces(xbins_size=1, selector=dict(type="histogram"))

            fig2.update_traces(marker_line_width=2)

            fig2.add_annotation(
                text="Your datapoint is here",
                showarrow=True,
                arrowhead=1,
                x=list(self.featuredict.values())[1],
                y=0.2,
                align="left",
                xanchor="left",
                font=dict(size=15, color="#242526"),
            )

            st.plotly_chart(fig2, use_container_width=True)

            selected_points2 = plotly_events(fig2)

            if selected_points2:

                xvalue = int(selected_points2[0]["x"])

                selectdf2 = selectdf[(selectdf[list(self.featuredict)[1]] == xvalue)]

                fig3 = px.histogram(
                    selectdf2,
                    x=list(self.featuredict)[2],
                    color="Model Performance",
                    category_orders={
                        "Model Performance": [
                            "High Risk-Predicted Incorrectly",
                            "High Risk-Predicted Correctly",
                            "Low Risk-Predicted Incorrectly",
                            "Low Risk-Predicted Correctly",
                        ]
                    },
                )

                fig3.update_traces(marker_line_width=2)

                fig3.add_annotation(
                    text="Your datapoint is here",
                    showarrow=True,
                    arrowhead=1,
                    x=list(self.featuredict.values())[2],
                    y=0.2,
                    align="left",
                    xanchor="left",
                    font=dict(size=15, color="#242526"),
                )

                 st.plotly_chart(fig3, use_container_width=True)

                st.plotly_chart(fig3)

    def interactiveexp(self, idnum):
        col1, col2 = st.columns(2)

        with col1:
            # selectmode = st.radio(
            #     "You can choose to explore:",
            #     [
            #         "Feature Distribution(select one feature)",
            #         "Feature Interaction(select two features)",
            #     ],
            #     key="Feature Distribution",
            # )

            self.featurebar(idnum)

            if self.click:

                with col2:
                    # st.write(" ")
                    # st.write(" ")
                    # st.write(" ")
                    # st.write(" ")
                    # st.write(" ")
                    # st.write(" ")

    
                    category_orders = [
                            "High Risk-Predicted Incorrectly",
                            "High Risk-Predicted Correctly",
                            "Low Risk-Predicted Incorrectly",
                            "Low Risk-Predicted Correctly",
                        ]

                    colors_dict = {
                            "High Risk-Predicted Incorrectly": "rgb(161,40,48)",
                            "High Risk-Predicted Correctly": "rgb(240,17,0)",
                            "Low Risk-Predicted Incorrectly": "rgb(143,151,121)",
                            "Low Risk-Predicted Correctly": "rgb(79,121,66)",
                        }
                    plotly_colors = [colors_dict[c] for c in category_orders]

                    fig1 = px.histogram(
                            self.train,
                            x=self.select,
                            color="Model Performance",
                            category_orders={
                                "Model Performance": [
                                    "High Risk-Predicted Incorrectly",
                                    "High Risk-Predicted Correctly",
                                    "Low Risk-Predicted Incorrectly",
                                    "Low Risk-Predicted Correctly",
                                ]
                            },
                            color_discrete_sequence=plotly_colors,
                        )
                        # fig1.add_annotation(
                        #     text='Your datapoint is here',
                        #     showarrow=True,
                        #     arrowhead=1,
                        #     x=self.featuredict[self.select],
                        #     y=0,
                        #     align="center",
                        #     xanchor="left",
                        #     yanchor="bottom",
                        #     ay=180,
                        #     font=dict(size=15, color="#242526"),
                        # )

                    fig1.update_traces(marker_line_width=2)

                    fig1.update_layout(legend=dict(y=-0.3, orientation="h"))

                    fig1.update_layout(height=500)

                    st.plotly_chart(fig1, use_container_width=True,height =500)

                    self.verbalexp(idnum)


if __name__ == "__main__":

    a = Model()
    a.test_model()
    # a.showconfusion(a.train.RiskPerformance, a.train.Predict, 1)
    a.showmodelperf(a.train.RiskPerformance, a.train.Predict, 1)

    # testid = [54]
    # ,54,411,192,463,528,384,395,408,417

    idnum = Newapp

    # for idnum in Newapp:
    
    verbalpred = "The model predicts the selected application has *high* risk and recommends you *reject* it." if a.test.iloc[idnum]["Predict"] == 1 else "The model predicts the selected application has *low* risk and recommends you *accept* it."


    st.subheader(verbalpred)
    
    # st.write("The reality is ", a.test.iloc[idnum]["RiskPerformance"])

    a.shapfeatureimportance(idnum)
        # a.plotdisforimportantfeature(idnum)
    a.importfeaturedict(idnum)


    st.write('**Why did our machine learning make this prediction?**')
    st.write('Our machine learning model is trained on many previous applications for which whether the applicants repay the loan is known.')
    st.write("Our model has learned from these applications. Each feature of the application can increase or decrease the applicant's chance of repaying, depending on the value of the feature")


    a.interactiveexp(idnum)
