import glob,os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from fancyimpute import KNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv('Bank_CS.csv')
df = data.copy()
continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
categorical = [col for col in df.columns if col not in continuous]


menu = ["Raw Dataset","Data Preprocessing","EDA","Correlation Analysis","Feature Selection","SMOTE","Default Modeling","Clustering","Model Tuning","Result of Model Tuning","Model Visualization"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Raw Dataset":
        st.title("Raw Data")
        number = st.number_input("Number of Rows to View",1,3000,5)
        st.dataframe(df.head(number))

        st.title("Choose Column")
        if st.checkbox("Select Columns to Show"):
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect("Select",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

        st.title("Basic Infomation on raw data")
        st.write(df.dtypes)

        st.title("Shape of dataset")
        st.write(df.shape)

        
        

elif choice == "Data Preprocessing":
        st.title("Data Preprocessing")
        EDA = ["1. Drop Unnessasary Column","2. Checking cleanliness of data","3. Setting Categorical and Numerical data","4. Checking the percentage of missing values","5. KNN Imputing"
        ,"6. Binning Numerical Data","7. Label Encoding","8. Association Rule Mining"]
        EDA = st.sidebar.radio("EDA",EDA)
        if EDA == "1. Drop Unnessasary Column":
                st.subheader("Droped Unnamed:0 , Unnamed 0.1")
                df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
                number = st.number_input("Number of Rows to View",1,3000,5)
                st.dataframe(df.head(number))
        elif EDA == "2. Checking cleanliness of data":
                st.subheader("Data Cleaning")
                objectType = df.select_dtypes(include=['object']).copy()
                numType = df.select_dtypes(include=['float64','int64']).copy()
                for i,col in enumerate(objectType):
                        df[col] = df[col].str.lower()
                for i,col in enumerate(objectType):
                        st.write(col + ' : '  + str(objectType[col].unique()))
                st.subheader("As we can see, the state have many form.")
                st.subheader("Exp: Johor -- Johor B , Pulau Penang -- P.Pinang")
                st.subheader("Therefore, we need to clean all these data with issued.")
                st.header("Below are the catogorical result after cleaning.")
                df['State'] = df['State'].replace(dict.fromkeys(['johor','johor b'], 'Johor'))
                df['State'] = df['State'].replace(dict.fromkeys(['selangor'], 'Selangor'))
                df['State'] = df['State'].replace(dict.fromkeys(['kuala lumpur','k.l'], 'Kuala Lumpur'))
                df['State'] = df['State'].replace(dict.fromkeys(['penang','p.pinang','pulau penang'], 'Penang'))
                df['State'] = df['State'].replace(dict.fromkeys(['n.sembilan','n.s'], 'Negeri Sembilan'))
                df['State'] = df['State'].replace(dict.fromkeys(['sarawak','swk'], 'Sarawak'))
                df['State'] = df['State'].replace(dict.fromkeys(['sabah'], 'Sabah'))
                df['State'] = df['State'].replace(dict.fromkeys(['kedah'], 'Kedah'))
                df['State'] = df['State'].replace(dict.fromkeys(['trengganu'], 'Terengganu'))

                for col in objectType.columns:
                        st.write(col + ' : '  + str(df[col].unique()))
                st.header("Below are the numerical result.")

                for i,col in enumerate(numType):
                        st.write(col + ' : '  + str(numType[col].unique()))
        
        elif EDA == "3. Setting Categorical and Numerical data":
                df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
                df['State'] = df['State'].replace(dict.fromkeys(['johor','johor b'], 'Johor'))
                df['State'] = df['State'].replace(dict.fromkeys(['selangor'], 'Selangor'))
                df['State'] = df['State'].replace(dict.fromkeys(['kuala lumpur','k.l'], 'Kuala Lumpur'))
                df['State'] = df['State'].replace(dict.fromkeys(['penang','p.pinang','pulau penang'], 'Penang'))
                df['State'] = df['State'].replace(dict.fromkeys(['n.sembilan','n.s'], 'Negeri Sembilan'))
                df['State'] = df['State'].replace(dict.fromkeys(['sarawak','swk'], 'Sarawak'))
                df['State'] = df['State'].replace(dict.fromkeys(['sabah'], 'Sabah'))
                df['State'] = df['State'].replace(dict.fromkeys(['kedah'], 'Kedah'))
                df['State'] = df['State'].replace(dict.fromkeys(['trengganu'], 'Terengganu'))
                st.subheader("Numerical DataFrame")
                continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
                number = st.number_input("Number of Rows to View",1,3000,5)
                st.dataframe(df[continuous].head(number))
                st.subheader("Show Numerical Column")
                if st.checkbox("Numerical Columns"):
                        all_columns = continuous
                        selected_columns = st.multiselect("Select",all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df)

                st.subheader("Show Categorical Data")
                categorical = [col for col in df.columns if col not in continuous]
                cat_df = st.number_input("Number of Rows to View",1,2500,5)
                st.dataframe(df[categorical].head(cat_df))
                st.subheader("Show Categorical Column")
                if st.checkbox("Select Categorical Columns"):
                        all_columns = categorical
                        selected_columns = st.multiselect("Select",all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df)

        elif EDA == "4. Checking the percentage of missing values":
                df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
                df['State'] = df['State'].replace(dict.fromkeys(['johor','johor b'], 'Johor'))
                df['State'] = df['State'].replace(dict.fromkeys(['selangor'], 'Selangor'))
                df['State'] = df['State'].replace(dict.fromkeys(['kuala lumpur','k.l'], 'Kuala Lumpur'))
                df['State'] = df['State'].replace(dict.fromkeys(['penang','p.pinang','pulau penang'], 'Penang'))
                df['State'] = df['State'].replace(dict.fromkeys(['n.sembilan','n.s'], 'Negeri Sembilan'))
                df['State'] = df['State'].replace(dict.fromkeys(['sarawak','swk'], 'Sarawak'))
                df['State'] = df['State'].replace(dict.fromkeys(['sabah'], 'Sabah'))
                df['State'] = df['State'].replace(dict.fromkeys(['kedah'], 'Kedah'))
                df['State'] = df['State'].replace(dict.fromkeys(['trengganu'], 'Terengganu'))
                
                nulls = df.columns[df.isna().any()].tolist()
                st.title("Percentage of Missing Values")
                for i,col in enumerate(nulls):
                        st.write('Percent of missing ' + col + ' records is ' + str((df[col].isnull().sum()/df.shape[0])*100) + '%' )
        
        elif EDA == "5. KNN Imputing":
                st.title("KNN Imputing")
                df = df.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
                df['State'] = df['State'].replace(dict.fromkeys(['johor','johor b'], 'Johor'))
                df['State'] = df['State'].replace(dict.fromkeys(['selangor'], 'Selangor'))
                df['State'] = df['State'].replace(dict.fromkeys(['kuala lumpur','k.l'], 'Kuala Lumpur'))
                df['State'] = df['State'].replace(dict.fromkeys(['penang','p.pinang','pulau penang'], 'Penang'))
                df['State'] = df['State'].replace(dict.fromkeys(['n.sembilan','n.s'], 'Negeri Sembilan'))
                df['State'] = df['State'].replace(dict.fromkeys(['sarawak','swk'], 'Sarawak'))
                df['State'] = df['State'].replace(dict.fromkeys(['sabah'], 'Sabah'))
                df['State'] = df['State'].replace(dict.fromkeys(['kedah'], 'Kedah'))
                df['State'] = df['State'].replace(dict.fromkeys(['trengganu'], 'Terengganu'))
                st.header("Since KNN Imputing can only work on numerical values, we need to first encode the object datatypes.")
                objectDf = df.select_dtypes(include=['object']).copy()
                objectCols = objectDf.columns
                st.write("Object Type DataFrame Before Encoding")
                st.write(objectDf)

                from sklearn.preprocessing import OrdinalEncoder

                tempEncoded = df.copy()
                encoder = OrdinalEncoder()
                imputer = KNN()
                # List of encoder object
                encoderList = []
                for col in objectCols:
                        #'''function to encode non-null data and replace it in the original data'''
                        #retains only non-null values
                        nonulls = np.array(tempEncoded[col].dropna())
                        #reshapes the data for encoding
                        impute_reshape = nonulls.reshape(-1,1)
                        #print(impute_reshape)
                        #encode date
                        encoder = OrdinalEncoder()
                        encoder.fit(impute_reshape)
                        encoderList.append(encoder)
                        #print(encoder.categories_)
                        impute_ordinal = encoder.transform(impute_reshape)
                        #encoderCategories.append(encoder.categories_)
                        #Assign back encoded values to non-null values
                        tempEncoded[col].loc[tempEncoded[col].notnull()] = np.squeeze(impute_ordinal)
                st.write("After Encoding Object Type Data Frame")
                st.write(tempEncoded)
                st.write("")
                tempEncoded = pd.DataFrame(np.round(imputer.fit_transform(tempEncoded)),columns = tempEncoded.columns)
                st.subheader("Then, we will start imputing the missing value.")
                st.write("")
                st.write("Below are the dataframe after imputing using KNN")
                st.write(tempEncoded)
                st.write("Decoded Data Frame")
                filledDf = tempEncoded.copy()
                for i,col in enumerate(objectCols):
                        filledDf[col] = filledDf[col].astype(object)
                        filledDf[col] = encoderList[i].inverse_transform(np.array(tempEncoded[col]).reshape(-1,1))
                st.write(filledDf)

        elif EDA == "6. Binning Numerical Data":
                filledDf = pd.read_csv('filledDf.csv')
                filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
                st.title("Binning Numerical Data")
                # Binning Joined Salary 
                st.subheader("Before Binning Numerical Data")
                st.write(filledDf)

                bins = [0,11627.75,15497.75,20000]
                groups = ["Low_Joined",'Medium_Joined',"High_Joined"]
                filledDf['Joined_Income_Binned'] = pd.cut(filledDf['Total_Income_for_Join_Application'], bins,labels=groups)
                filledDf['Joined_Income_Binned'] = filledDf['Joined_Income_Binned'].astype(object)

                # Binning Salary
                bins = [0,6095.75,9908.75,12562.00]
                groups = ["Low_Income",'Medium_Income',"High_Income"]
                filledDf['Income_Binned'] = pd.cut(filledDf['Monthly_Salary'], bins,labels=groups)
                filledDf['Income_Binned'] = filledDf['Income_Binned'].astype(object)

                #Binning Loan Amount
                bins = [0,305179.50,586650.25,799628.00]
                groups = ["Small_Loan",'Medium_Loan',"High_Loan"]
                filledDf['Loan_Amount_Binned'] = pd.cut(filledDf['Loan_Amount'], bins,labels=groups)
                filledDf['Loan_Amount_Binned'] = filledDf['Loan_Amount_Binned'].astype(object)

                #Binning Total Sum Of Loan
                bins = [0,727947.2, 1176512 ,	1500000]
                groups = ["Small_Loan_Sum",'Medium_Loan_Sum',"High_Loan_Sum"]
                filledDf['Total_Loan_Sum_Binned'] = pd.cut(filledDf['Total_Sum_of_Loan'], bins,labels=groups)
                filledDf['Total_Loan_Sum_Binned'] = filledDf['Total_Loan_Sum_Binned'].astype(object)

                # Adding categorical columns to the categorical list
                continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
                categorical = [col for col in filledDf.columns if col not in continuous]

                st.subheader("After Binning Numerical Data")
                st.write(filledDf)
                
                for cat in categorical:
                        filledDf[cat] = filledDf[cat].astype(object)
                
                cols = list(filledDf.columns.values) #Make a list of all of the columns in the df
                cols.pop(cols.index('Decision'))
                for con in continuous:
                        cols.pop(cols.index(con))
                        filledDf = filledDf[cols+ continuous +['Decision']]

                st.write("After Rearranging Columns and format data types")
                st.write(filledDf)

        elif EDA == "7. Label Encoding":
                filledDf = pd.read_csv('beforelabelecoding.csv')
                filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
                continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
                categorical = [col for col in filledDf.columns if col not in continuous]
                st.header("DataFrame Before Label Encoding")
                st.write(filledDf)
                from sklearn.preprocessing import LabelEncoder
                from collections import defaultdict
                d = defaultdict(LabelEncoder)
                encodedDf = filledDf.copy()
                encodedDf[categorical] = filledDf[categorical].apply(lambda x: d[x.name].fit_transform(x))
                st.header("DataFrame After Label Encoding")
                st.write(encodedDf)
                
        elif EDA == "8. Association Rule Mining":
                st.title("Association Rule Mining")
                filledDf = pd.read_csv('filledDf.csv')
                filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
                encodedDf = pd.read_csv('encodedDf.csv')
                encodedDf = encodedDf.drop(['Unnamed: 0'],axis=1)
                continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
                categorical = [col for col in filledDf.columns if col not in continuous]
                from apyori import apriori
                from mlxtend.frequent_patterns import apriori
                from mlxtend.frequent_patterns import association_rules

                st.header("One Hot Encoding")
                OHEDf = pd.get_dummies(filledDf[categorical],columns=categorical)
                st.write(OHEDf)

                st.header("Apriori")
                frequent_itemsets = apriori(OHEDf, min_support=0.5, use_colnames=True)
                st.write(frequent_itemsets)

                st.header("Association Rule Mining Result")
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
                st.write(rules)

elif choice == "EDA":
        st.title("Exploratory Data Analysis")
        filledDf = pd.read_csv('filledDf.csv')
        filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
        encodedDf = pd.read_csv('encodedDf.csv')
        encodedDf = encodedDf.drop(['Unnamed: 0'],axis=1)
        continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
        categorical = [col for col in filledDf.columns if col not in continuous]
        
        st.header("1. What types of properties do couples of different income group go for?")
        fig, ax = plt.subplots(figsize=(10,7))
        sns.countplot(data=filledDf,x='Joined_Income_Binned',hue='Property_Type',ax=ax)
        st.pyplot()

        st.header("2. Which type of employee are most likely accepted by the bank for the loan?")
        fig, ax = plt.subplots(figsize=(10,7))
        sns.countplot(data=filledDf[filledDf['Decision']=='accept'],x='Employment_Type',ax=ax)
        st.pyplot()

        st.header("3. Which are the distributions of high,medium and low income in different states?")
        fig, ax = plt.subplots(figsize=(12,7))
        sns.barplot(data=filledDf,x='State',y='Loan_Amount',ax=ax)
        st.pyplot()

        st.header("4. Which state has the highest number of loan approval and rejects?")
        fig, ax = plt.subplots(figsize=(12,7))
        sns.countplot(data=filledDf,x='State',hue='Decision',ax=ax)
        st.pyplot()
        
        st.header("5. Which group of people applies the most loans and get rejected/accepted?")
        fig, ax = plt.subplots(figsize=(12,7))
        sns.barplot(data=filledDf,x='Employment_Type',y='Loan_Amount', hue='Decision',ax=ax)
        st.pyplot()

        st.header("6. Do people with higher credit score who apply for higher loan amount get more approvals?")
        fig, ax = plt.subplots(figsize=(12,7))
        sns.stripplot(data=filledDf,x='Score',y='Loan_Amount',jitter=0.05,hue='Decision',dodge=True,ax=ax)
        st.pyplot()

        st.header("7. What is the relationship between joined income and sum of loan taken?")
        sns.set(style="ticks")
        fig, ax = plt.subplots(figsize=(15,7))
        ax = sns.scatterplot(x="Total_Income_for_Join_Application", y="Total_Sum_of_Loan",hue='Joined_Income_Binned', data=filledDf)
        st.pyplot()
        st.write("The middle classes seems to have more instances of loan application.")

        st.header("8. What is the relationship between monthly salary and joined income?")
        sns.set(style="ticks")
        fig, ax = plt.subplots(figsize=(15,7))
        ax = sns.scatterplot(x="Monthly_Salary", y="Total_Income_for_Join_Application",hue='Joined_Income_Binned', data=filledDf)
        st.pyplot()
        st.write("Regardless of monthly salary, total joined income from the middle class seems to be the most.")

elif choice == "Correlation Analysis":
        encodedDf = pd.read_csv('encodedDf.csv')
        encodedDf = encodedDf.drop(['Unnamed: 0'],axis=1)
        corr = encodedDf.corr()

        # generating a 
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(17, 15))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        st.title("Correlation Analysis")
        sns.heatmap(corr, mask=mask, cmap=cmap,vmin=-0.5,vmax=.5, center=0, annot=True,fmt=".2f",
                square=True, linewidths=.5, cbar_kws={"shrink": .5},ax=ax)
        st.pyplot()
        

elif choice == "Feature Selection":
        filledDf = pd.read_csv('filledDf.csv')
        filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
        encodedDf = pd.read_csv('encodedDf.csv')
        encodedDf = encodedDf.drop(['Unnamed: 0'],axis=1)
        continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
        categorical = [col for col in filledDf.columns if col not in continuous]

        from sklearn.ensemble import RandomForestClassifier
        from boruta import BorutaPy
        from sklearn.feature_selection import RFECV
        from sklearn.preprocessing import MinMaxScaler

        def ranking(ranks, names, order=1):
                minmax = MinMaxScaler()
                ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
                ranks = map(lambda x: round(x,2), ranks)
                return dict(zip(names, ranks))
        
        X = encodedDf.drop('Decision',1)
        y = encodedDf.Decision
        colnames = X.columns
        
        X = encodedDf.drop(columns='Decision').copy()
        y = encodedDf['Decision'].copy()
        
        st.title("Boruta")
        rf = RandomForestClassifier(n_jobs=1, class_weight="balanced", max_depth=5,random_state=42)
        feat_selector = BorutaPy(rf, n_estimators="auto", random_state=1)

        feat_selector.fit(X.values,y.values.ravel())
        colnames = X.columns

        boruta_score = ranking(list(map(float, feat_selector.ranking_)),colnames, order=-1)
        boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
        boruta_score = boruta_score.sort_values("Score", ascending=False)

        st.write('---------Top 10----------')
        st.write(boruta_score.head(10))

        st.write('---------Bottom 10 ----------')
        st.write(boruta_score.tail(10))

        sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:26], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
        plt.title("Boruta Top Related Features in Ascending Orders")
        st.pyplot()

        st.title("RFE")
        rf = RandomForestClassifier(n_jobs=1, class_weight="balanced", max_depth=5, n_estimators=100)
        rf.fit(X,y)
        rfe = RFECV(rf, min_features_to_select=1, cv=3)

        rfe.fit(X,y)

        rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
        rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
        rfe_score = rfe_score.sort_values("Score", ascending = False)

        sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score[0:26], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
        plt.title("RFE Top Related Features in Ascending Order")
        st.pyplot()

        st.write('---------Top 10----------')
        st.write(rfe_score.head(10))

        st.write('---------Bottom 10----------')
        st.write(rfe_score.tail(10))

        st.title("Selecting Top Feature")
        st.subheader('Top Features for Boruta are : \n')
        topFeaturesBoruta = boruta_score.loc[boruta_score['Score'] > 0.5,'Features'].values
        for i,feature in enumerate(topFeaturesBoruta):
                st.write(i+1,feature)

        
        st.subheader('Top Features for RFE are : \n')
        topFeaturesRFE = rfe_score.loc[rfe_score['Score'] > 0.5,'Features'].values
        for i,feature in enumerate(topFeaturesRFE):
                st.write(i+1,feature)

        topMatchedFeatures = []
        for i in topFeaturesBoruta:
                for j in topFeaturesRFE:
                        if i==j:
                                topMatchedFeatures.append(i)

        # Printing Matched Features
        st.subheader('Matched Features are : \n')
        for i,feature in enumerate(topMatchedFeatures):
                st.write(i+1,feature)

elif choice == "SMOTE":
        st.title("SMOTE")
        filledDf = pd.read_csv('filledDf.csv')
        filledDf = filledDf.drop(['Unnamed: 0'],axis=1)
        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)
        continuous = ['Loan_Amount','Monthly_Salary','Total_Sum_of_Loan','Total_Income_for_Join_Application']
        categorical = [col for col in encodedTopFeaturesDf.columns if col not in continuous]

        st.header("Before SMOTE, Lets normalize the encoded top features dataset")
        st.write(encodedTopFeaturesDf)
        temp = encodedTopFeaturesDf[continuous]
        columns = encodedTopFeaturesDf[continuous].columns

        st.header("After Normalizing")
        a = MinMaxScaler()
        d = defaultdict(LabelEncoder)
        encodedDf = filledDf.copy()
        encodedDf[categorical] = filledDf[categorical].apply(lambda x: d[x.name].fit_transform(x))
        x_scaled = a.fit_transform(temp)
        x_scaled = pd.DataFrame(x_scaled, columns = columns)
        x_scaled = x_scaled.round(2)
        #Forming the scaled dataframe
        encodedTopFeaturesDf[continuous] = x_scaled
        st.write(encodedTopFeaturesDf)
        st.write(encodedTopFeaturesDf.shape)

        st.subheader("Before SMOTE, Lets Decode the topMatchFeatures Data Frame to perform SMOTENC")
        topFeaturesCategorical = encodedTopFeaturesDf.columns.intersection(categorical)
        topFeaturesCategorical = topFeaturesCategorical.drop(labels=['Decision'])

        topFeaturesDf = encodedTopFeaturesDf.copy()
        topFeaturesDf[topFeaturesCategorical] = encodedTopFeaturesDf[topFeaturesCategorical].apply(lambda x: d[x.name].inverse_transform(x))
        topFeaturesDf.info()

        from imblearn.over_sampling import SMOTENC
        X = topFeaturesDf.drop('Decision',1)
        y = topFeaturesDf.Decision
        smtNC = SMOTENC(categorical_features=[topFeaturesDf.dtypes==object],sampling_strategy="not majority", random_state=42, k_neighbors=5)
        X_res, y_res = smtNC.fit_resample(X, y)
        X_res = pd.DataFrame(X_res,columns = X.columns)
        y_res = pd.DataFrame(y_res)
        smotedDf = pd.concat([X_res,y_res], axis=1)
        smotedDf.columns = [*smotedDf.columns[:-1], 'Decision']
        st.title("SMOTED Data Frame")
        st.write(smotedDf)
        st.write(smotedDf.shape)

        for con in continuous:
                smotedDf[con] = smotedDf[con].astype(float)

        st.title("Encode Back the SMOTED Data Frame")
        encodedSmotedDf = smotedDf.copy()
        encodedSmotedDf[topFeaturesCategorical] = smotedDf[topFeaturesCategorical].apply(lambda x: d[x.name].fit_transform(x))
        st.write(encodedSmotedDf)
        st.write(encodedSmotedDf.shape)

elif choice == "Default Modeling":
        classifier = ["Decision Tree Classifier","Random Forest Classifier","Support Vector Machine","Naive Bayes","MLP Classifier"]
        classifier = st.sidebar.selectbox("Classifier",classifier)
        encodedSmotedDf = pd.read_csv('encodedSmotedDf.csv')
        encodedSmotedDf = encodedSmotedDf.drop(['Unnamed: 0'],axis=1)

        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)

        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

        s_fprList = []
        s_tprList = []

        ns_fprList = []
        ns_tprList = []

        ns_X = encodedTopFeaturesDf.drop('Decision',1)
        ns_y = encodedTopFeaturesDf.Decision
        ns_X_train,ns_X_test,ns_y_train,ns_y_test = train_test_split(ns_X,ns_y,test_size=0.3,random_state =42)

        s_X = encodedSmotedDf.drop('Decision',1)
        s_y = encodedSmotedDf.Decision
        s_X_train,s_X_test,s_y_train,s_y_test = train_test_split(s_X,s_y,test_size=0.3,random_state =42)
        if classifier == "Decision Tree Classifier":
                st.title("Decision Tree Classifier")
                smoted_dt = ["Non-Smote-d","Smote-d"]
                smoted_dt = st.sidebar.radio("SMOTED",smoted_dt)

                def treeClassifier(X_train,X_test,y_train,y_test,criterion,depth,leaf):
                        tree_clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf =leaf,random_state=42)
                        tree_clf.fit(X_train,y_train)
                        y_pred = tree_clf.predict(X_test)
                        st.write("Accuracy on training set : {:.3f}".format(tree_clf.score(X_train, y_train)))
                        st.write("Accuracy on test set     : {:.3f}".format(tree_clf.score(X_test, y_test)))
                        confusion_majority=confusion_matrix(y_test, y_pred)

                        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

                        st.write('**********************')
                        st.write('Mjority TN = ', confusion_majority[0][0])
                        st.write('Mjority FP = ', confusion_majority[0][1])
                        st.write('Mjority FN = ', confusion_majority[1][0])
                        st.write('Mjority TP = ', confusion_majority[1][1])
                        st.write('**********************')

                        st.write('Precision= {:.3f}'.format(precision_score(y_test, y_pred)))
                        st.write('Recall= {:.3f}'. format(recall_score(y_test, y_pred)))
                        st.write('F1= {:.3f}'. format(f1_score(y_test, y_pred)))
                        st.write('Accuracy= {:.3f}'. format(accuracy_score(y_test, y_pred)))

                        prob_DT = tree_clf.predict_proba(X_test)
                        prob_DT = prob_DT[:,1]

                        auc_DT = roc_auc_score(y_test,prob_DT)
                        st.write("AUC : %.2f " % auc_DT)
                        fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT)

                        return fpr_DT,tpr_DT
                
                if smoted_dt == "Non-Smote-d":
                        fpr,tpr = treeClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,'gini',16,2)
                        ns_fprList.append(fpr)
                        ns_tprList.append(tpr)

                elif smoted_dt == "Smote-d":
                        fpr,tpr = treeClassifier(s_X_train,s_X_test,s_y_train,s_y_test,'gini',19,1)
                        s_fprList.append(fpr)
                        s_tprList.append(tpr)

        elif classifier == "Random Forest Classifier":
                st.title("Random Forest Classifier")
                smoted_rf = ["Non-Smote-d","Smote-d"]
                smoted_rf = st.sidebar.radio("SMOTED",smoted_rf)

                def forestClassifier(X_train,X_test,y_train,y_test,criterion,depth,leaf,estimators):
                        rf = RandomForestClassifier(criterion=criterion,max_depth=depth,min_samples_leaf=leaf,n_estimators=estimators,random_state=42)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        st.write("Accuracy on training set : {:.3f}".format(rf.score(X_train, y_train)))
                        st.write("Accuracy on test set     : {:.3f}".format(rf.score(X_test, y_test)))
                        confusion_majority=confusion_matrix(y_test, y_pred)

                        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

                        st.write('**********************')
                        st.write('Mjority TN = ', confusion_majority[0][0])
                        st.write('Mjority FP = ', confusion_majority[0][1])
                        st.write('Mjority FN = ', confusion_majority[1][0])
                        st.write('Mjority TP = ', confusion_majority[1][1])
                        st.write('**********************')

                        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
                        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
                        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
                        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))



                        prob_rf = rf.predict_proba(X_test)
                        prob_rf = prob_rf[:,1]

                        auc_rf = roc_auc_score(y_test,prob_rf)
                        st.write("AUC : %.2f " % auc_rf)
                        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, prob_rf)
                        return fpr_rf,tpr_rf

                if smoted_rf == "Non-Smote-d":
                        fpr,tpr =forestClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,'gini',16,2,140)
                        ns_fprList.append(fpr)
                        ns_tprList.append(tpr)

                elif smoted_rf == "Smote-d":
                        fpr,tpr = forestClassifier(s_X_train,s_X_test,s_y_train,s_y_test,'gini',19,1,100)
                        s_fprList.append(fpr)
                        s_tprList.append(tpr)

        elif classifier == "Support Vector Machine":
                st.title("Support Vector Machine")
                smoted_svm = ["Non-Smote-d","Smote-d"]
                smoted_svm = st.sidebar.radio("SMOTED",smoted_svm)

                def svmClassifier(X_train,X_test,y_train,y_test,C,gamma,kernel):
                        svc = SVC( C=C,gamma=gamma,kernel=kernel,probability=True,random_state=42)
                        svc.fit(X_train, y_train)
                        y_pred = svc.predict(X_test)
                        st.write("Accuracy on training set : {:.3f}".format(svc.score(X_train, y_train)))
                        st.write("Accuracy on test set     : {:.3f}".format(svc.score(X_test, y_test)))
                        confusion_majority=confusion_matrix(y_test, y_pred)

                        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

                        st.write('**********************')
                        st.write('Mjority TN = ', confusion_majority[0][0])
                        st.write('Mjority FP = ', confusion_majority[0][1])
                        st.write('Mjority FN = ', confusion_majority[1][0])
                        st.write('Mjority TP = ', confusion_majority[1][1])
                        st.write('**********************')

                        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
                        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
                        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
                        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

                        prob_svc = svc.predict_proba(X_test)
                        prob_svc = prob_svc[:,1]

                        auc_svc = roc_auc_score(y_test,prob_svc)
                        st.write("AUC : %.2f " % auc_svc)
                        fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, prob_svc)
                        return fpr_svc,tpr_svc

                if smoted_svm == "Non-Smote-d":
                        fpr,tpr = svmClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,0.7,1.0,'rbf')
                        ns_fprList.append(fpr)
                        ns_tprList.append(tpr)

                elif smoted_svm == "Smote-d":
                        fpr,tpr = svmClassifier(s_X_train,s_X_test,s_y_train,s_y_test,0.7,1.0,'rbf')
                        s_fprList.append(fpr)
                        s_tprList.append(tpr)

        elif classifier == "Naive Bayes":
                st.title("Naive Bayes")
                smoted_nb = ["Non-Smote-d","Smote-d"]
                smoted_nb = st.sidebar.radio("SMOTED",smoted_nb)

                def naiveBayesClassifier(X_train,X_test,y_train,y_test):
                        nb = GaussianNB()
                        nb.fit(X_train, y_train)
                        y_pred = nb.predict(X_test)
                        st.write("Accuracy on training set : {:.3f}".format(nb.score(X_train, y_train)))
                        st.write("Accuracy on test set     : {:.3f}".format(nb.score(X_test, y_test)))
                        confusion_majority=confusion_matrix(y_test, y_pred)

                        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

                        st.write('**********************')
                        st.write('Mjority TN = ', confusion_majority[0][0])
                        st.write('Mjority FP = ', confusion_majority[0][1])
                        st.write('Mjority FN = ', confusion_majority[1][0])
                        st.write('Mjority TP = ', confusion_majority[1][1])
                        st.write('**********************')

                        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
                        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
                        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
                        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

                        prob_NB = nb.predict_proba(X_test)
                        prob_NB = prob_NB[:,1]

                        auc_NB= roc_auc_score(y_test, prob_NB)
                        st.write('AUC : %.2f' % auc_NB)

                        fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB) 
                        return fpr_NB, tpr_NB

                if smoted_nb == "Non-Smote-d":
                        fpr,tpr = naiveBayesClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test)
                        ns_fprList.append(fpr)
                        ns_tprList.append(tpr)

                elif smoted_nb == "Smote-d":
                        fpr,tpr = naiveBayesClassifier(s_X_train,s_X_test,s_y_train,s_y_test)
                        s_fprList.append(fpr)
                        s_tprList.append(tpr)

        elif classifier == "MLP Classifier":
                st.title("MLP Classifier")
                smoted_mlp = ["Non-Smote-d","Smote-d"]
                smoted_mlp = st.sidebar.radio("SMOTED",smoted_mlp)
                from sklearn.neural_network import MLPClassifier
                def mlpClassifier(X_train,X_test,y_train,y_test):
                        mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=42)
                        mlp.fit(X_train, y_train)
                        y_pred = mlp.predict(X_test)

                        st.write("Accuracy on test set     : {:.3f}".format(mlp.score(X_test, y_test)))
                        st.write("Accuracy on test set     : {:.3f}".format(mlp.score(X_test, y_test)))
                        confusion_majority=confusion_matrix(y_test, y_pred)

                        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

                        st.write('**********************')
                        st.write('Mjority TN = ', confusion_majority[0][0])
                        st.write('Mjority FP = ', confusion_majority[0][1])
                        st.write('Mjority FN = ', confusion_majority[1][0])
                        st.write('Mjority TP = ', confusion_majority[1][1])
                        st.write('**********************')

                        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred)))
                        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred)))
                        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred)))
                        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

                        prob_mlp = mlp.predict_proba(X_test)
                        prob_mlp = prob_mlp[:,1]

                        auc_mlp= roc_auc_score(y_test, prob_mlp)
                        print('AUC : %.2f' % auc_mlp)

                        fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, prob_mlp) 
                        return fpr_mlp, tpr_mlp
                if smoted_mlp == "Non-Smote-d":
                        fpr,tpr = mlpClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test)

                elif smoted_mlp == "Smote-d":
                        fpr,tpr = mlpClassifier(s_X_train,s_X_test,s_y_train,s_y_test)

        

elif choice == "Clustering":
        encodedSmotedDf = pd.read_csv('encodedSmotedDf.csv')
        encodedSmotedDf = encodedSmotedDf.drop(['Unnamed: 0'],axis=1)

        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)
        from kmodes.kmodes import KModes
        kModeDf = encodedSmotedDf.drop(columns=continuous)
        km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
        fitClusters_cao = km_cao.fit_predict(kModeDf)
        st.title("K-Mode Clustering")
        clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
        clusterCentroidsDf.columns = kModeDf.columns
        cost = []
        for num_clusters in list(range(1,5)):
                kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
                kmode.fit_predict(kModeDf)
                cost.append(kmode.cost_)
        y = np.array([i for i in range(1,5,1)])
        plt.plot(y,cost)
        st.pyplot()

        kModeDf = encodedSmotedDf.drop(columns=continuous).reset_index()
        kModeClustersDf = pd.DataFrame(fitClusters_cao)
        kModeClustersDf.columns = ['kMode_Clusters']
        encodedKModeDf = pd.concat([kModeDf, kModeClustersDf], axis = 1).reset_index()
        encodedKModeDf = encodedKModeDf.drop(['index', 'level_0'], axis = 1)
        st.title("Encoded K-Mode Clustering Data Frame")
        st.write(encodedKModeDf)

        st.title("KMODE Visualization")
        import matplotlib.image as mpimg
        kmode1 = mpimg.imread('kmode1.PNG')
        plt.axis('off')
        plt.imshow(kmode1)
        st.pyplot()
        kmode1 = mpimg.imread('kmode2.PNG')
        plt.axis('off')
        plt.imshow(kmode1)
        st.pyplot()
        kmode1 = mpimg.imread('kmode3.PNG')
        plt.axis('off')
        plt.imshow(kmode1)
        st.pyplot()

        from sklearn.cluster import KMeans
        st.title("K-Means")
        X = encodedSmotedDf[continuous].copy()
        wcss = []
        for i in range(1, 11):
                kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
                kmeans.fit(X)
                # inertia method returns wcss for that model
                wcss.append(kmeans.inertia_)
        plt.figure(figsize=(10,5))
        sns.lineplot(range(1, 11), wcss,marker='o',color='red')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
        st.pyplot()     

        st.title("Encoded K-Means Clustering Data Frame")
        kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
        y_kmeans = kmeans.fit_predict(X)

        kMeansDf = encodedSmotedDf[continuous].reset_index()
        kMeansClustersDf = pd.DataFrame(y_kmeans)
        kMeansClustersDf.columns = ['kMeans_Clusters']
        encodedKMeansDf = pd.concat([kMeansDf, kMeansClustersDf], axis = 1).reset_index()
        encodedKMeansDf = encodedKMeansDf.drop(['index', 'level_0'], axis = 1)
        st.write(encodedKMeansDf)

elif choice == "Model Tuning":
        from sklearn.model_selection import GridSearchCV
        encodedSmotedDf = pd.read_csv('encodedSmotedDf.csv')
        encodedSmotedDf = encodedSmotedDf.drop(['Unnamed: 0'],axis=1)

        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)

        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

        ns_X = encodedTopFeaturesDf.drop('Decision',1)
        ns_y = encodedTopFeaturesDf.Decision
        ns_X_train,ns_X_test,ns_y_train,ns_y_test = train_test_split(ns_X,ns_y,test_size=0.3,random_state =42)

        s_X = encodedSmotedDf.drop('Decision',1)
        s_y = encodedSmotedDf.Decision
        s_X_train,s_X_test,s_y_train,s_y_test = train_test_split(s_X,s_y,test_size=0.3,random_state =42)

        s_fprList = []
        s_tprList = []

        ns_fprList = []
        ns_tprList = []

        def model_tuning(estimator, param_grid, X_train, y_train):
                grid = GridSearchCV(estimator,param_grid,cv=5)
                grid.fit(X_train, y_train)
                st.write('\n\n******************************')
                st.write('The best parameters are %s with CV score of %0.2f' 
                        % (grid.best_params_,grid.best_score_))
                st.write('****************************')

        st.title("Decision Tree Classifier")
        param_grid = {
                'max_depth' : np.arange(1,20,2),
                'min_samples_leaf' : np.arange(1,10,1),
                'criterion':['gini','entropy']
                }
        model_tuning(DecisionTreeClassifier(random_state=42),
                        param_grid,
                        s_X_train,
                        s_y_train)

        st.title("Random Forest Claissifier")
        param_grid = {
                'n_estimators' : np.arange(100,150,5),
                }
        model_tuning(RandomForestClassifier(criterion='gini', max_depth=19, min_samples_leaf = 1,random_state=42),
                        param_grid,
                        s_X_train,
                        s_y_train)

        st.title("Support Vector Machine")
        param_grid = {
                'C':[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1],
                'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                'kernel':['rbf','linear']
                }
        model_tuning(SVC(random_state=42),
                param_grid,
                s_X_train,
                s_y_train
                )
        
        st.title("Logistic Regression")
        
        param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'max_iter' : np.arange(100,500,25)
                }

        model_tuning(LogisticRegression(random_state=42),
                        param_grid,
                        s_X_train,
                        s_y_train)

             
elif choice == "Result of Model Tuning":
        classifier_name = ["Decision Tree Classifier","Random Forest Classifier","Support Vector Machine","Logistic Regression"]
        classifier_name = st.sidebar.selectbox("Classifier",classifier_name)
        encodedSmotedDf = pd.read_csv('encodedSmotedDf.csv')
        encodedSmotedDf = encodedSmotedDf.drop(['Unnamed: 0'],axis=1)

        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)

        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

        ns_X = encodedTopFeaturesDf.drop('Decision',1)
        ns_y = encodedTopFeaturesDf.Decision
        ns_X_train,ns_X_test,ns_y_train,ns_y_test = train_test_split(ns_X,ns_y,test_size=0.3,random_state =42)

        s_X = encodedSmotedDf.drop('Decision',1)
        s_y = encodedSmotedDf.Decision
        s_X_train,s_X_test,s_y_train,s_y_test = train_test_split(s_X,s_y,test_size=0.3,random_state =42)

        def add_parameter_ui(clf_name):
                params = dict()
                if clf_name == "Decision Tree Classifier":
                        max_dep = st.sidebar.slider("Max Depth",1,30,15)
                        params["Max Depth"] = max_dep

                        min_sam_l = st.sidebar.slider("Minimum Sample Leaf",1,50,4)
                        params["Minimum Sample Leaf"] = min_sam_l

                elif clf_name =="Random Forest Classifier":
                        n_estimators = st.sidebar.slider("n_estimators",100,150,135)
                        params["n_estimators"] = n_estimators

                elif clf_name == "Support Vector Machine":
                        C = st.sidebar.slider("c",1,10,1)
                        params["C"] = C

                        gamma = st.sidebar.slider("gamma",1,10,1)
                        params["gamma"] = gamma
                else:
                        C = st.sidebar.slider("C",1,10,1)
                        params["C"] = C
                return params

        params = add_parameter_ui(classifier_name)
        st.write(params)

        def get_classifier(clf_name,params):
                if clf_name == "Decision Tree Classifier":
                        st.title("Decision Tree Classifier")
                        clf = DecisionTreeClassifier(criterion='gini',max_depth=params["Max Depth"],min_samples_leaf = params["Minimum Sample Leaf"],random_state=42)
                elif clf_name =="Random Forest Classifier":
                        st.title("Random Forest Classifier")
                        clf = RandomForestClassifier(random_state=42,n_estimators=params["n_estimators"])
                elif clf_name =="Support Vector Machine":
                        st.title("Support Vector Machine")
                        clf = SVC(C=params["C"], kernel="linear",gamma=params["gamma"],probability=True,random_state=42)
                else:
                        clf = LogisticRegression(C=params["C"])
                return clf

        clf = get_classifier(classifier_name,params)


        #Classification
        s_X = encodedSmotedDf.drop('Decision',1)
        s_y = encodedSmotedDf.Decision
        s_X_train,s_X_test,s_y_train,s_y_test = train_test_split(s_X,s_y,test_size=0.3,random_state =42)

        clf.fit(s_X_train,s_y_train)
        y_pred = clf.predict(s_X_test)
        st.write(f"classifier = {classifier_name}")
        st.write("Accuracy on training set : {:.3f}".format(clf.score(s_X_train, s_y_train)))
        st.write("Accuracy on test set     : {:.3f}".format(clf.score(s_X_test, s_y_test)))
        confusion_majority=confusion_matrix(s_y_test, y_pred)

        st.write('Mjority classifier Confusion Matrix\n', confusion_majority)

        st.write('**********************')
        st.write('Mjority TN = ', confusion_majority[0][0])
        st.write('Mjority FP = ', confusion_majority[0][1])
        st.write('Mjority FN = ', confusion_majority[1][0])
        st.write('Mjority TP = ', confusion_majority[1][1])
        st.write('**********************')

        st.write('Precision= {:.3f}'.format(precision_score(s_y_test, y_pred)))
        st.write('Recall= {:.3f}'. format(recall_score(s_y_test, y_pred)))
        st.write('F1= {:.3f}'. format(f1_score(s_y_test, y_pred)))
        st.write('Accuracy= {:.3f}'. format(accuracy_score(s_y_test, y_pred)))

        prob_DT = clf.predict_proba(s_X_test)
        prob_DT = prob_DT[:,1]

        auc_DT = roc_auc_score(s_y_test,prob_DT)
        st.write("AUC : %.2f " % auc_DT)
        fpr_DT, tpr_DT, thresholds_DT = roc_curve(s_y_test, prob_DT)

else:
        st.title("Model Visualization")
        encodedSmotedDf = pd.read_csv('encodedSmotedDf.csv')
        encodedSmotedDf = encodedSmotedDf.drop(['Unnamed: 0'],axis=1)

        encodedTopFeaturesDf = pd.read_csv('encodedTopFeaturesDf.csv')
        encodedTopFeaturesDf = encodedTopFeaturesDf.drop(['Unnamed: 0'],axis=1)

        from sklearn.model_selection import train_test_split 
        from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

        s_fprList = []
        s_tprList = []

        ns_fprList = []
        ns_tprList = []

        ns_X = encodedTopFeaturesDf.drop('Decision',1)
        ns_y = encodedTopFeaturesDf.Decision
        ns_X_train,ns_X_test,ns_y_train,ns_y_test = train_test_split(ns_X,ns_y,test_size=0.3,random_state =42)

        s_X = encodedSmotedDf.drop('Decision',1)
        s_y = encodedSmotedDf.Decision
        s_X_train,s_X_test,s_y_train,s_y_test = train_test_split(s_X,s_y,test_size=0.3,random_state =42)

        def treeClassifier(X_train,X_test,y_train,y_test,criterion,depth,leaf):
                tree_clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, min_samples_leaf =leaf,random_state=42)
                tree_clf.fit(X_train,y_train)
                y_pred = tree_clf.predict(X_test)
                prob_DT = tree_clf.predict_proba(X_test)
                prob_DT = prob_DT[:,1]

                auc_DT = roc_auc_score(y_test,prob_DT)
                fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test, prob_DT)

                return fpr_DT,tpr_DT
        fpr,tpr = treeClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,'gini',16,2)
        ns_fprList.append(fpr)
        ns_tprList.append(tpr)
        fpr,tpr = treeClassifier(s_X_train,s_X_test,s_y_train,s_y_test,'gini',19,1)
        s_fprList.append(fpr)
        s_tprList.append(tpr)

        def forestClassifier(X_train,X_test,y_train,y_test,criterion,depth,leaf,estimators):
                rf = RandomForestClassifier(criterion=criterion,max_depth=depth,min_samples_leaf=leaf,n_estimators=estimators,random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                prob_rf = rf.predict_proba(X_test)
                prob_rf = prob_rf[:,1]

                auc_rf = roc_auc_score(y_test,prob_rf)
                fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, prob_rf)
                return fpr_rf,tpr_rf
        fpr,tpr =forestClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,'gini',16,2,140)
        ns_fprList.append(fpr)
        ns_tprList.append(tpr)
        fpr,tpr = forestClassifier(s_X_train,s_X_test,s_y_train,s_y_test,'gini',19,1,100)
        s_fprList.append(fpr)
        s_tprList.append(tpr)
        
        def svmClassifier(X_train,X_test,y_train,y_test,C,gamma,kernel):
                svc = SVC( C=C,gamma=gamma,kernel=kernel,probability=True,random_state=42)
                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)
                prob_svc = svc.predict_proba(X_test)
                prob_svc = prob_svc[:,1]

                auc_svc = roc_auc_score(y_test,prob_svc)
                fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, prob_svc)
                return fpr_svc,tpr_svc
        
        fpr,tpr = svmClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test,0.7,1.0,'rbf')
        ns_fprList.append(fpr)
        ns_tprList.append(tpr)


        fpr,tpr = svmClassifier(s_X_train,s_X_test,s_y_train,s_y_test,0.7,1.0,'rbf')
        s_fprList.append(fpr)
        s_tprList.append(tpr)
        
        def naiveBayesClassifier(X_train,X_test,y_train,y_test):
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                y_pred = nb.predict(X_test)

                prob_NB = nb.predict_proba(X_test)
                prob_NB = prob_NB[:,1]

                auc_NB= roc_auc_score(y_test, prob_NB)
                fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_NB) 
                return fpr_NB, tpr_NB

        fpr,tpr = naiveBayesClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test)
        ns_fprList.append(fpr)
        ns_tprList.append(tpr)
        
        fpr,tpr = naiveBayesClassifier(s_X_train,s_X_test,s_y_train,s_y_test)
        s_fprList.append(fpr)
        s_tprList.append(tpr)

        from sklearn.neural_network import MLPClassifier
        def mlpClassifier(X_train,X_test,y_train,y_test):
                mlp = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=42)
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)

                prob_mlp = mlp.predict_proba(X_test)
                prob_mlp = prob_mlp[:,1]

                fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, prob_mlp) 
                return fpr_mlp, tpr_mlp

        fpr,tpr = mlpClassifier(ns_X_train,ns_X_test,ns_y_train,ns_y_test)
        ns_fprList.append(fpr)
        ns_tprList.append(tpr)

        fpr,tpr = mlpClassifier(s_X_train,s_X_test,s_y_train,s_y_test)
        s_fprList.append(fpr)
        s_tprList.append(tpr)

        def modelComparison(fprList,tprList,labelList,colorList,title):
                plt.figure(figsize=(15,10))
                for i in range(len(labelList)):
                        plt.plot(fprList[i], tprList[i] , color=colorList[i], label=labelList[i]) 
                plt.plot([0, 1], [0, 1], color='black', linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.suptitle(title, fontsize=30)
                plt.legend()
                st.pyplot()
        
        modelComparison(ns_fprList,ns_tprList,['DT','RFC','SVM','NB','MLP'],['orange','blue','red','yellow','green'],'Non-Smote-d (ROC) Curve')
        modelComparison(s_fprList,s_tprList,['DT','RFC','SVM','NB','MLP'],['orange','blue','red','yellow','green'],'Smote-d (ROC) Curve')