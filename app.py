import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

mpl.rcParams['font.size'] = 10
mpl.rcParams['text.color'] = 'black'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import chi2, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index as c_idx

background_color = '#0E1117'

st.sidebar.title('Navigation')
page = st.sidebar.radio('What would you like to do', ['Predictive Model', 'Survival Model'])

if page == 'Predictive Model':
    row0col1, row0col2 = st.columns([1, 5])
    with row0col1:
        st.image('1249-heart-beat-outline.gif')
    with row0col2:
        st.title('Heart Failure Prediction')

    # file = st.file_uploader('Insert Data')
    file = True

    if file:
        # data = pd.read_csv(file.name)
        data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

        labelled_data = data.copy()
        labelled_data['anaemia'] = labelled_data['anaemia'].map({0: 'No', 1: 'Yes'})
        labelled_data['diabetes'] = labelled_data['diabetes'].map({0: 'No', 1: 'Yes'})
        labelled_data['high_blood_pressure'] = labelled_data['high_blood_pressure'].map({0: 'No', 1: 'Yes'})
        labelled_data['sex'] = labelled_data['sex'].map({0: 'Female', 1: 'Male'})
        labelled_data['smoking'] = labelled_data['smoking'].map({0: 'No', 1: 'Yes'})
        labelled_data['DEATH_EVENT'] = labelled_data['DEATH_EVENT'].map({0: 'Alive', 1: 'Dead'})

        dependent_var = st.selectbox('Select the Dependent Variable', data.columns, index=len(data.columns)-1)
        no_dependent_var = st.checkbox('Dependent Variable not available ?')

        st.header('Independent Variables')
        independent_var = data.drop(dependent_var, axis=1)
        row0a, row0b, row0c = [], [], []
        row0a.extend(st.columns([1, 1, 1.5, 1]))
        row0b.extend(st.columns(4))
        row0c.extend(st.columns(4))
        for i in range(4):
            row0a[i].write(independent_var.columns[i])
            row0b[i].write(independent_var.columns[i + 4])
            row0c[i].write(independent_var.columns[i + 8])

        if not no_dependent_var:
            st.header('Dependent Variable')
            st.subheader(dependent_var)
            st.write('The label here considered is DEATH_EVENT which depicts the death of a patient during the '
                     'follow-up period of the treatment.')

        # discrete vs continuous features
        discrete_features, continuous_features = [], []
        for feature in data.columns:
            if feature == dependent_var:
                continue
            elif len(data[feature].unique()) >= 10 and feature != 'time':
                continuous_features.append(feature)
            elif feature != 'time':
                discrete_features.append(feature)
        # print('Discrete: ', discrete_features, '\n', 'Continuous', continuous_features)

        row1col1, row1col2 = st.columns(2)
        with row1col1:
            st.header('Discrete Features')
            for feature in discrete_features:
                st.write(feature)
        with row1col2:
            st.header('Continuous Features')
            for feature in continuous_features:
                st.write(feature)

        def autopct_format(values):
            def my_format(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return '{v:d}'.format(v=val)

            return my_format

        # pie plot
        st.header('Counts and Count Difference')
        row2col1, row2col2 = st.columns([1, 2.5])
        with row2col1:
            st.subheader('Select a feature')
            if not no_dependent_var:
                option1 = st.radio('', discrete_features + [dependent_var])
            else:
                option1 = st.radio('', discrete_features)
        with row2col2:
            if option1 == dependent_var:
                st.subheader(option1 + ' - Dependent Variable')

                fig = go.Figure(data=[go.Pie(labels=list(Counter(labelled_data[option1]).keys()),
                                             values=list(Counter(labelled_data[option1]).values()),
                                             name='')])
                # plt.pie(Counter(labelled_data[option1]).values(), labels=Counter(labelled_data[option1]).keys(),
                #         startangle=90, autopct='%1.1f%%', radius=1.5,
                #         wedgeprops={'edgecolor': 'black', 'linewidth': 1})
            else:
                st.subheader(option1)
                # st.write(feature_details[option1])

                fig = go.Figure(data=[go.Pie(labels=list(Counter(labelled_data[option1]).keys()),
                                     values=list(Counter(labelled_data[option1]).values()),
                                     name='')])
                # plt.pie(Counter(labelled_data[option1]).values(), labels=Counter(labelled_data[option1]).keys(),
                #         startangle=90, autopct='%1.1f%%', radius=1.5,
                #         wedgeprops={'edgecolor': 'black', 'linewidth': 1})

            fig.update_traces(textfont_size=20)
            fig.update_layout(height=400, width=400, plot_bgcolor='rgb(245,245,240)', font=dict(size=16))
            st.write(fig)

        # subplot pie chart
        if not no_dependent_var and option1 != dependent_var:
            # fig, ax = plt.subplots(1, 2, figsize=(6, 4))
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]])

            transformed_data = labelled_data.groupby([option1, 'DEATH_EVENT']).size().reset_index(name='count')
            for i, u in enumerate(labelled_data[option1].unique()):
                # ax[i].title.set_text(option1 + ' - ' + str(u))
                fig.add_trace(go.Pie(labels=transformed_data[transformed_data[option1] == u]['DEATH_EVENT'],
                                     values=transformed_data[transformed_data[option1] == u]['count'],
                                     title=option1 + ' - ' + str(u),
                                     name=''), row=1, col=i+1)
                # ax[i].pie(transformed_data[transformed_data[option1] == u]['count'],
                #           labels=transformed_data[transformed_data[option1] == u]['DEATH_EVENT'], startangle=90,
                #           autopct='%1.1f%%')

            fig.update_traces(textfont_size=20)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), font=dict(size=16))
            st.write(fig)

        # correlation matrix
        st.header('Correlation Matrix')
        selected_features = st.multiselect('Features', data.columns[:-1])
        row3col1, row3col2 = st.columns([6, 1])
        correlate = 0
        with row3col1:
            select_all_features = st.checkbox('Select All Features')
        with row3col2:
            if st.button('Correlate'):
                correlate = 1
        if select_all_features:
            selected_features = list(data.drop(['time', dependent_var], axis=1).columns)
        if selected_features and (select_all_features or correlate):
            correlation = data[selected_features].corr()
            correlate = 0
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(correlation, cmap="Blues", vmax=0.6, annot=True, annot_kws={"size": 11}, fmt='.2f')
            st.pyplot(fig)

        # kde plot and histogram
        binned_data = data.copy()
        age_bins = [18, 45, 60, 80]
        binned_data['Age Bins'] = pd.cut(binned_data['age'], age_bins)
        st.header('Probability Density')
        row4col1, row4col2 = st.columns([1, 2])
        with row4col1:
            st.subheader('Select a feature')
            option2 = st.radio('', continuous_features)

            visualize_type = st.radio('', ['KDE Plot', 'Histogram'])
        with row4col2:
            st.subheader(option2)
            fig = plt.figure(figsize=(8, 6))
            plt.xticks(fontsize=12)
            if visualize_type == 'Histogram':
                sns.histplot(x=option2, data=data, bins=20)
            else:
                if no_dependent_var:
                    sns.kdeplot(x=option2, data=labelled_data, fill=True)
                else:
                    sns.kdeplot(x=option2, hue=dependent_var, data=labelled_data, fill=True)
            st.pyplot(fig)

        # st.write(feature_details[option2])

        # value distribution
        if not no_dependent_var:
            st.subheader('Value Distribution')
            binned_data = labelled_data.copy()
            bin_division = {
                'age_bins': [18, 45, 60, 80, 100],
                'creatinine_phosphokinase_bins': [0, 200, 2000, 5000, max(binned_data['creatinine_phosphokinase'])+1],
                'ejection_fraction_bins': [0, 20, 50, 75, 100],
                'platelets_bins': [10000, 150000, 450000, 1000000],
                'serum_creatinine_bins': [0, 0.5, 1.2, 5, 11],
                'serum_sodium_bins': [100, 130, 145, 150]
            }
            bin_labels = {
                'age_labels': ['18-45', '45-60', '60-80', '>80'],
                'creatinine_phosphokinase_labels': ['10-200 (N)', '200-2000', '2000-5000', '>5000'],
                'ejection_fraction_labels': ['0-20', '20-50', '50-75 (N)', '70-100'],
                'platelets_labels': ['10k-150k', '150k-450k (N)', '>450k'],
                'serum_creatinine_labels': ['0-0.5', '0.5-1.2 (N)', '1.2-5', '>5'],
                'serum_sodium_labels': ['100-130', '130-145', '145-150']
            }

            binned_data[option2+'_bins'] = pd.cut(binned_data[option2], bin_division[option2+'_bins'],
                                                  labels=bin_labels[option2+'_labels'])

            death_0_values = binned_data[binned_data['DEATH_EVENT'] == 'Alive']
            death_1_values = binned_data[binned_data['DEATH_EVENT'] == 'Dead']

            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                                subplot_titles=[option2.capitalize() + ' wrt Alive patients',
                                                option2.capitalize() + ' wrt Deceased patients'])

            fig.add_trace(go.Pie(labels=list(Counter(death_0_values[option2+'_bins']).keys()),
                                 values=list(Counter(death_0_values[option2+'_bins']).values()),
                                 name=''), row=1, col=1)

            fig.add_trace(go.Pie(labels=list(Counter(death_1_values[option2 + '_bins']).keys()),
                                 values=list(Counter(death_1_values[option2 + '_bins']).values()),
                                 name=''), row=1, col=2)

            fig.update_traces(textfont_size=20)
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), font=dict(size=16))
            st.write(fig)

            transformed_data = binned_data.groupby([option2+'_bins', 'DEATH_EVENT']).size().reset_index(name='count')

            input_data = {option2+' Groups': list(transformed_data[option2+'_bins'].unique())}
            for i in transformed_data['DEATH_EVENT'].unique():
                input_data['Count of ' + i + ' Patients'] = list(
                    transformed_data[transformed_data['DEATH_EVENT'] == i]['count'])

            counts_table = pd.DataFrame(input_data)
            # counts_table.set_index('Age Groups', inplace=True)

            counts_table['% of Alive Patients wrt ' + option2 + ' Group'] = round(counts_table['Count of Alive Patients']
                        * 100 / (counts_table['Count of Alive Patients'] + counts_table['Count of Dead Patients']), 2)
            counts_table['% of Dead Patients wrt ' + option2 + ' Group'] = round(counts_table['Count of Dead Patients']
                        * 100 / (counts_table['Count of Alive Patients'] + counts_table['Count of Dead Patients']), 2)

            counts_table['% of Alive Patients wrt Total Alive Patients'] = round(
                counts_table['Count of Alive Patients'] * 100 / counts_table['Count of Alive Patients'].sum(), 2)
            counts_table['% of Dead Patients wrt Total Dead Patients'] = round(
                counts_table['Count of Dead Patients'] * 100 / counts_table['Count of Dead Patients'].sum(), 2)

            st.table(counts_table)

        # feature selection
        st.header('Feature Importance')

        fig = plt.figure(figsize=(6, 4))
        importance = mutual_info_classif(data.drop(['time', 'DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
        feat_importance = pd.Series(importance, data.drop(['time', 'DEATH_EVENT'], axis=1).columns)
        feat_importance.plot(kind='barh', color='teal')
        st.pyplot(fig)

        best_features = SelectKBest(chi2, k=10)
        features_ranking = best_features.fit(data.drop(['DEATH_EVENT'], axis=1), data['DEATH_EVENT'])
        # features_ranking_df = pd.Series(features_ranking, data.columns[:-1])
        ranking_dictionary = {}
        for i in range(len(features_ranking.scores_)):
            ranking_dictionary[data.columns[i]] = round(features_ranking.scores_[i], 3)
        asc_sort = sorted(ranking_dictionary.items(), key=lambda kv: (kv[1], kv[0]))
        plot_data = {'features': [i for i, j in asc_sort], 'scores': list(np.arange(1, 13))}

        # prediction
        predict_button = st.button('Predict')
        predict_button = True
        if predict_button:
            if not no_dependent_var:
                X = data.copy()
                y = data[dependent_var]
                X_train, X_testi, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

                X_train_classify = X_train[['ejection_fraction', 'serum_creatinine', 'platelets']]
                y_train_classify = y_train
                X_testi_classify = X_testi[['ejection_fraction', 'serum_creatinine', 'platelets']]
                y_test_classify = y_test

                lof = LocalOutlierFactor()
                outlier_rows = lof.fit_predict(X_train_classify)

                mask = outlier_rows != -1
                X_train_classify, y_train_classify = X_train_classify[mask], y_train_classify[mask]

                oversample = RandomOverSampler(sampling_strategy='minority')
                X_train_classify, y_train_classify = oversample.fit_resample(X_train_classify, y_train_classify)

                scaler = StandardScaler()
                X_train_classify = scaler.fit_transform(X_train_classify)
                X_test_classify = scaler.transform(X_testi_classify)

                model = SVC(kernel='linear', gamma=0.1, probability=True)
                model.fit(X_train_classify, y_train_classify)
                y_pred = model.predict(X_test_classify)
                st.write('Accuracy: ' + str(accuracy_score(y_test_classify, y_pred)*100))
                st.write('Recall: ' + str(recall_score(y_test_classify, y_pred) * 100))
                # st.write(classification_report(y_test, y_pred))

                fig, ax = plt.subplots(figsize=(2, 2))
                fig.set_size_inches(2, 2, forward=True)
                conf_matrix = confusion_matrix(y_true=y_test_classify, y_pred=y_pred)
                ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='small')

                plt.xlabel('Predictions', fontsize=6)
                plt.ylabel('Actuals', fontsize=6)
                plt.title('Confusion Matrix', fontsize=6)
                st.pyplot(fig)

                # survival special part
                X_train_survival = X_train.drop(['platelets', 'creatinine_phosphokinase'], axis=1)
                X_test_survival = X_testi.drop(['platelets', 'creatinine_phosphokinase'], axis=1)

                lof = LocalOutlierFactor()
                outlier_rows = lof.fit_predict(X_train_survival)

                mask = outlier_rows != -1
                X_train_survival = X_train_survival[mask]

                scaler = StandardScaler()
                X_train_survival.values[:] = scaler.fit_transform(X_train_survival)

                cph = CoxPHFitter()
                # update the dynamic var
                cph.fit(X_train_survival, duration_col='time', event_col='DEATH_EVENT', step_size=0.01)
                hazard_score = cph.predict_partial_hazard(X_test_survival)
                hazard_score = pd.DataFrame({'Patient ID': hazard_score.index, 'Hazard Function Score': hazard_score.values})
                hazard_score['Hazard Function Score'] = round(hazard_score['Hazard Function Score'], 2)

                survival_score = cph.predict_survival_function(X_test_survival)
                survival_score = survival_score.T
                time_median = data['time'].median()
                summary_cols = [survival_score.columns[0], time_median, survival_score.columns[-1]]
                survival_score_summary = survival_score[summary_cols]
                survival_score_summary.reset_index(inplace=True)
                survival_score_summary.rename(columns={'index': 'Patient ID',
                                                       summary_cols[0]: 'Survival Probability at the start of Time Period',
                                                       summary_cols[1]: 'Survival Probability in the middle of Time Period',
                                                       summary_cols[2]: 'Survival Probability at the end of Time Period'}, inplace=True)

                predicted_data = pd.DataFrame(X_testi_classify, columns=['ejection_fraction', 'serum_creatinine',
                                                                        'platelets'])
                predicted_data.reset_index(inplace=True)
                predicted_data.rename(columns={'index': 'Patient ID'}, inplace=True)
                predicted_data['Risk Probability'] = 0

                for i in range(predicted_data.shape[0]):
                    prediction_values = np.array(X_test_classify[i]).reshape(1, -1)
                    predicted_data['Risk Probability'].iloc[i] = model.predict_proba(prediction_values)[:, 1]

                predicted_data = predicted_data.merge(hazard_score, on='Patient ID')
                predicted_data = predicted_data.merge(survival_score_summary, on='Patient ID')

                st.dataframe(predicted_data.sort_values(by=['Risk Probability'], ascending=False))

            elif no_dependent_var:  # cluster section
                st.subheader('Clustering Result')
                X = data.drop(['time', 'DEATH_EVENT'], axis=1)
                y = data['DEATH_EVENT']

                lof = LocalOutlierFactor()
                outlier_rows = lof.fit_predict(X)

                mask = outlier_rows != -1
                X, y = X[mask], y[mask]

                oversample = RandomOverSampler(sampling_strategy='minority')
                X, y = oversample.fit_resample(X, y)
                print(Counter(y))

                scaler = StandardScaler()
                X_final = scaler.fit_transform(X)

                cluster_model = KMeans(n_clusters=2, n_init=30, tol=0.00001, max_iter=1000)
                cluster_model.fit(X_final)
                y_pred = cluster_model.labels_
                labels = pd.DataFrame(cluster_model.labels_)
                X_new = pd.concat((X, labels), axis=1)
                X_new = X_new.rename({0: 'labels'}, axis=1)

                f1, f2 = st.columns(2)
                with f1:
                    feature1 = st.selectbox('Feature 1', X.columns)
                with f2:
                    feature2 = st.selectbox('Feature 2', X.columns, index=2)
                fig = plt.figure(figsize=(8, 6))
                plt.xticks(fontsize=12)
                sns.scatterplot(x=feature1, y=feature2, hue='labels', data=X_new)
                st.pyplot(fig)


    else:
        st.subheader('Please upload a file!')

if page == 'Survival Model':
    row0col1, row0col2 = st.columns([1, 5])
    with row0col1:
        st.image('1249-heart-beat-outline.gif')
    with row0col2:
        st.title('Heart Failure Survival Analysis')

    # file = st.file_uploader('Insert Data')
    file = True

    if file:
        # data = pd.read_csv(file.name)
        data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

        labelled_data = data.copy()
        labelled_data['anaemia'] = labelled_data['anaemia'].map({0: 'No', 1: 'Yes'})
        labelled_data['diabetes'] = labelled_data['diabetes'].map({0: 'No', 1: 'Yes'})
        labelled_data['high_blood_pressure'] = labelled_data['high_blood_pressure'].map({0: 'No', 1: 'Yes'})
        labelled_data['sex'] = labelled_data['sex'].map({0: 'Female', 1: 'Male'})
        labelled_data['smoking'] = labelled_data['smoking'].map({0: 'No', 1: 'Yes'})
        labelled_data['DEATH_EVENT'] = labelled_data['DEATH_EVENT'].map({0: 'Alive', 1: 'Dead'})

        event_var = st.selectbox('Select the Event Describing Variable', data.columns, index=len(data.columns) - 1)

        time_var = st.selectbox('Select the Time Variable', data.columns)

        X = data[['ejection_fraction', 'smoking', 'serum_sodium', 'diabetes', 'high_blood_pressure', 'serum_creatinine',
                  'anaemia', 'sex', 'age', 'time', 'DEATH_EVENT']]

        X_train, X_test = train_test_split(X, random_state=10)

        lof = LocalOutlierFactor()
        outlier_rows = lof.fit_predict(X_train)

        mask = outlier_rows != -1
        X_train = X_train[mask]

        scaler = StandardScaler()
        X_train.values[:] = scaler.fit_transform(X_train)
        X_test.values[:] = scaler.transform(X_test)

        cph = CoxPHFitter()
        cph.fit(X_train, duration_col='time', event_col='DEATH_EVENT', step_size=0.01) # update the dynamic var

        hazard_score = cph.predict_partial_hazard(X_test)
        hazard_score = pd.DataFrame({'Patient ID': hazard_score.index, 'Risk Value': hazard_score.values})
        hazard_score.set_index('Patient ID', inplace=True)
        hazard_score['Risk Value'] = round(hazard_score['Risk Value'], 2)
        hazard_score.sort_values(by=['Risk Value'], ascending=False, inplace=True)

        row1col1, row1col2 = st.columns([1, 1])
        with row1col1:
            st.subheader('Hazard Function')
            st.write('The Hazard Function shows the chances of death of the patient due to Heart Failure. The table '
                     'alongside depicts the patients with highest risk of death.')

        with row1col2:
            st.dataframe(hazard_score[:10])

        st.subheader('Survival Function')
        st.write('Survival Function calculates survival probabilities of the patient over the course of time. The value'
                 ' lies between 0 to 1. The graph represents the survival probability of patients with highest risk.')
        patient_id = hazard_score[:10].index
        survival_score = cph.predict_survival_function(X_test)
        top_least_survivals = {}
        for i in patient_id:
            top_least_survivals[i] = survival_score[i].values

        tls_df = pd.DataFrame(top_least_survivals, index=survival_score.index)
        fig, ax = plt.subplots(1, figsize=(4, 2))
        ax.set_ylim(bottom=0)
        for i in tls_df.columns:
            ax.plot(tls_df.index, tls_df[i])

        st.pyplot(fig)

        more_details_button = st.button('More Details')
        if more_details_button:
            more_details = []
            for i in patient_id:
                details = [i]
                details.extend(labelled_data.loc[i].values)
                more_details.append(details)

            more_details_df = pd.DataFrame(more_details, columns=['Patient ID'] + list(labelled_data.columns))
            st.dataframe(more_details_df)

