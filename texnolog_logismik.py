import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import umap 
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

#Φορτονουμε δεδομέμα
st.sidebar.title("Ανεβασε το αρχειο σου εδω!")
file = st.sidebar.file_uploader("Να είναι αρχείο CSV, TSV ή EXCEL", type=["csv", "xlsx", "tsv"])

#Συναρτηση για διάβασμα φορτωμένων αρχείων
def load_data(file):
    if file is not None:
        
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=";", header=0)  # Χρησιμοποιούμε την 1η γραμμή ως header
            st.write("Νέα προεπισκόπηση δεδομένων:", df.head())
            st.write("Στήλες dataset:", list(df.columns))

        elif file.name.endswith('.xlsx'):
            df= pd.read_excel(file, header=None) 
            df = df.dropna(how='all')#Διαγραφη κενων γραμμών
            for i,row in df.iterrows():
                if all(pd.notna(row)):
                    df.columns = row
                    df = df[i+1:].reset_index(drop=True)
                    break
        
        elif file.name.endswith('.tsv'):
            df= pd.read_csv(file, sep='\t')
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        return df
    return None

data = load_data(file)
if data is not None:
    st.write("Εδω βλέπεις το αρχείο που ανέβασες", data.head())
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_columns:
        data[col] = data[col].astype('category').cat.codes
    st.session_state['data']=data

# Τα Tabs
tabs =st.tabs(["Visualization tab", "Feature Selection", "Classification", "Results", "Info"])

#Το Tab του Visualization
with tabs[0]:
    if data is not None:
        st.subheader("EDA")

        fig, ax = plt.subplots()
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        fig2 = px.histogram(data, x=data.columns[0], title="Κατανομη!")
        st.plotly_chart(fig2)

        st.subheader("2D και 3D Οπτικοποίηση με PCA & UMAP")
        imputer = SimpleImputer(strategy='mean')
        data_cleaned = data.dropna(axis=1, how="all")  # Αφαιρούμε στήλες που είναι εξ ολοκλήρου NaN
        st.write("Προεπισκόπηση δεδομένων πριν από την επεξεργασία:")
        st.write(data_cleaned.head())
        st.write("Τύποι δεδομένων:")
        st.write(data_cleaned.dtypes)
        st.write("Μέγεθος δεδομένων πριν το Imputer:", data_cleaned.shape)
        features = imputer.fit_transform(data_cleaned.iloc[:, :-1])
        labels = data.iloc[:, -1].iloc[:features.shape[0]]  # Εξασφαλίζουμε σωστή διάσταση
        dim_reduction = st.selectbox("Διάλεξε αλγόριθμο", ["PCA", "UMAP"])
        n_components = st.slider("Διαλαξε παράμετρο", 2, 3, 2)
        reducer = PCA(n_components=n_components) if dim_reduction == "PCA" else umap.UMAP(n_components=n_components)
        reduced_data = reducer.fit_transform(features)
        
        if n_components == 2:
            fig3 = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=labels.astype(str))
        else:
            fig3 = px.scatter_3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], color=labels.astype(str))
        st.plotly_chart(fig3)

#Tabs Feature selection
with tabs[1]:
    if data is not None:
        st.subheader("Feature Selection")
        num_features = st.slider("Διαλεξε αριθμό για τα Top features", 1, data.shape[1] -1, 5)
        selector = SelectKBest(f_classif, k=num_features)
        x_new = selector.fit_transform(features, labels)
        reduced_data = pd.DataFrame(x_new, columns=[str(col) for col in range(x_new.shape[1])])
        reduced_data["Label"] = labels[:x_new.shape[0]]
        st.session_state["reduced_features"] = reduced_data
        st.write("Μειωμένο σετ Feature", reduced_data.head())

#Tab Classification
with tabs[2]:
    if data is not None:
        st.subheader("Classification πριν & μετά το Feature Selection")
        classifiers = {"KNN": KNeighborsClassifier, "Random Forest": RandomForestClassifier}
        model_choice = st.selectbox("Διάλεξε Αλγόριθμο", list(classifiers.keys()))
        param = st.number_input("Ορισμός παραμέτρου", min_value=1, value=3)
        
        def train_and_evaluate(X, y, model_choice, param):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = classifiers[model_choice](n_neighbors=param) if model_choice == "KNN" else classifiers[model_choice](n_estimators=param)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = "N/A"
            if hasattr(clf,"Predict_probab") and len(np.unique(y_test)) > 1:
                roc_auc = roc_auc_score(label_binarize(y_test, classes=np.unique(y_test)), clf.predict_proba(X_test), multi_class='ovr')
            return acc, f1, roc_auc
        

        # Αφαίρεση γραμμών όπου η στήλη-στόχος περιέχει NaN
        st.session_state["reduced_features"].dropna(subset=[st.session_state["reduced_features"].columns[-1]], inplace=True)
        acc1, f1_1, roc_auc1 = train_and_evaluate(features, labels, model_choice, param)
        if "reduced_features" in st.session_state:
            acc2, f1_2, roc_auc2 = train_and_evaluate(st.session_state["reduced_features"].iloc[:, :-1], st.session_state["reduced_features"].iloc[:, -1], model_choice, param)
        else:
            acc2, f1_2, roc_auc2 = "N/A", "N/A", "N/A"

        st.session_state["Results"] = {"Original": (acc1, f1_1, roc_auc1), "Reduced": (acc2, f1_2, roc_auc2)}
        
        st.write("Αποτελέσματα Ταξινόμησης στο αρχικό σύνολο δεδομένων:")
        st.write(f"Accuracy: {acc1:.2f}")
        st.write(f"F1-score: {f1_1:.2f}")
        st.write(f"ROC-AUC: {roc_auc1}")

        if "reduced_features" in st.session_state:
            st.write("Αποτελέσματα Ταξινόμησης στο μειωμένο σύνολο δεδομένων:")
            st.write(f"Accuracy: {acc2:.2f}")
            st.write(f"F1-score: {f1_2:.2f}")
            st.write(f"ROC-AUC: {roc_auc2}")

        
#Tabs Results
with tabs[3]:
    if "Results" in st.session_state and st.session_state["Results"]:
        st.subheader("Συγκριση Αποτελεσματων")
        results_df = pd.DataFrame(st.session_state["Results"], index =["Accuracy", "F1-Score", "ROC-AUC"])
        st.write(results_df)
        fig4 = px.bar(results_df, barmode="group", title="Συγκριση επιδοσεων")
        st.plotly_chart(fig4)

#Tab για το info
with tabs[4]:
    st.subheader("Application Info")
    st.write("Αυτη η εφαρμογη δημιουργηθηκε για εκπαιδευτικους σκοπους με χρηση Streamlit για data analysis και machine learning")
    st.write("Προγραμματίστικε εξ'ολοκλήρου απο τον: Γεώργιο-Φοίβο Αθανασόπουλο με αριθμό μητρώου: Π2020143, Φοιτητή του Ιονίου Πανεπηστημίου")
    st.write("Αρχικοποίηση απαραίτητων βιβλιοθηκών ωστε να μας βοηθήσουν στο υπολογισμό,εμφάνυση και ανάλυση των ανεβασμένων αρχείων...") 
    st.write("Μια συνάρτηση για το upload των tabular data...Μια μεταβλητή για δημιουργία κάθε σελίδας...Τελος λογικός προγραμματισμος σε κάθε σελίδα για την εμφάνηση του αποτελέσματος που θέλουμε!!!")

