import streamlit as st
import pandas as pd
import io
import warnings

# Pour éviter les messages d'avertissement
warnings.filterwarnings('ignore')

# Charger les données avec cache pour améliorer les performances
@st.cache_data
def load_data():
    urls = {
        "etablissement": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/base_etablissement_par_tranche_effectif.csv',
        "geographic": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/name_geographic_information.csv',
        "salaire": 'https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/net_salary_per_town_categories.csv'
    }
    etablissement = pd.read_csv(urls['etablissement'], sep=',')
    geographic = pd.read_csv(urls['geographic'], sep=',')
    salaire = pd.read_csv(urls['salaire'], sep=',')
    return etablissement, geographic, salaire

etablissement, geographic, salaire = load_data()

# Pré-traitement des données salaire
new_column_names_salaire = {
    'SNHM14': 'salaire',
    'SNHMC14': 'salaire_cadre',
    'SNHMP14': 'salaire_cadre_moyen',
    'SNHME14': 'salaire_employe',
    'SNHMO14': 'salaire_travailleur',
    'SNHMF14': 'salaire_femme',
    'SNHMFC14': 'salaire_cadre_femme',
    'SNHMFP14': 'salaire_cadre_moyen_femme',
    'SNHMFE14': 'salaire_employe_femme',
    'SNHMFO14': 'salaire_travailleur_femme',
    'SNHMH14': 'salaire_homme',
    'SNHMHC14': 'salaire_cadre_homme',
    'SNHMHP14': 'salaire_cadre_moyen_homme',
    'SNHMHE14': 'salaire_employe_homme',
    'SNHMHO14': 'salaire_travailleur_homme',
    'SNHM1814': 'salaire_18-25',
    'SNHM2614': 'salaire_26-50',
    'SNHM5014': 'salaire_+50',
    'SNHMF1814': 'salaire_18-25_femme',
    'SNHMF2614': 'salaire_26-50_femme',
    'SNHMF5014': 'salaire_+50_femme',
    'SNHMH1814': 'salaire_18-25_homme',
    'SNHMH2614': 'salaire_26-50_homme',
    'SNHMH5014': 'salaire_+50_homme'
}

salaire = salaire.rename(columns=new_column_names_salaire)
salaire['CODGEO'] = salaire['CODGEO'].str.lstrip('0').str.replace('A', '0').str.replace('B', '0')

# Configuration de la barre latérale
st.sidebar.title("Sommaire")
pages = ["👋 Intro", "🔍 Exploration des données", "📊 Data Visualisation", "🧩 Modélisation", "🔮 Prédiction", "📌 Conclusion"]
page = st.sidebar.radio("Aller vers", pages)

# Affichage de la sélection des données uniquement pour la page "Exploration des données"
if page == pages[1]:
    # Gestion de l'état de la page via session_state
    if 'page' not in st.session_state:
        st.session_state.page = "Etablissement"

    # Sélection de la page de données
    data_pages = ["Etablissement", "Geographic", "Salaire"]
    # st.sidebar.markdown("### Choix des données")
    st.session_state.page = st.sidebar.selectbox("Sélection du Dataframe", data_pages, index=data_pages.index(st.session_state.page))

st.sidebar.markdown(
    """
    - **Cursus** : Data Analyst
    - **Formation** : Formation Continue
    - **Mois** : Janvier 2024
    - **Groupe** : 
        - Christophe MONTORIOL
        - Issam YOUSR
        - Gwilherm DEVALLAN
        - Yacine OUDMINE
    """
)

# Définition des styles
st.markdown("""
    <style>
        h1 {color: #4629dd; font-size: 70px;}
        h2 {color: #440154ff; font-size: 50px;}
        h3 {color: #27dce0; font-size: 30px;}
        body {background-color: #f4f4f4;}
    </style>
""", unsafe_allow_html=True)

# Page d'introduction
if page == pages[0]:
    st.header("👋 Intro")
    st.caption("""**Cursus** : Data Analyst | **Formation** : Formation Continue | **Mois** : Janvier 2024 | **Groupe** : Christophe MONTORIOL, Issam YOUSR, Gwilherm DEVALLAN, Yacine OUDMINE""")
     # Ajouter l'image du bandeau
    st.image('https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/Bandeau_FrenchIndustry.png', use_column_width=True)
    st.write("""
        AAAL’objectif premier de ce projet est d’observer et de comprendre quelles sont les inégalités salariales en France. 
        À travers plusieurs jeux de données et plusieurs variables (géographiques, socio-professionnelles, démographiques, mais aussi du nombre d’entreprises par zone), 
        il sera question dans ce projet de mettre en lumière les facteurs d’inégalités les plus déterminants et de recenser ainsi les variables qui ont un impact significatif sur les deltas de salaire.
        En plus de distinguer les variables les plus déterminantes sur les niveaux de revenus, l’objectif de cette étude sera de construire des clusters ou des groupes de pairs basés sur les niveaux de salaire similaires.
        Enfin, un modèle de Machine Learning sera créé pour prédire au mieux un salaire en fonction des variables disponibles dans les jeux de données.
    """)

# Page d'exploration des données
elif page == pages[1]:
    st.header("🔍 Exploration des Données")

    # Fonction pour afficher les informations des DataFrames
    def afficher_info(dataframe, name):
        st.write(f"### {name}")
        st.write("#### Aperçu")
        st.write(dataframe.head())
        
        st.write("#### Informations")
        buffer = io.StringIO()
        dataframe.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.write("#### Statistiques")
        st.write(dataframe.describe())

    # Affichage des informations en fonction de la page sélectionnée
    if st.session_state.page == "Etablissement":
        afficher_info(etablissement, "Etablissement")
    elif st.session_state.page == "Geographic":
        afficher_info(geographic, "Geographic")
    elif st.session_state.page == "Salaire":
        afficher_info(salaire, "Salaire")

# Page de Data Visualisation
elif page == pages[2]:
    st.header("📊 Data Visualisation")

AAAAAAAAAAAAA

# Page de Modélisation
elif page == pages[3]:
    st.header("🧩 Modélisation")

# Page de Prédiction
elif page == pages[4]:
    st.header("🔮 Prédiction")
    st.subheader('Simulation de Prédiction avec Random Forest Regressor')

# Page de Conclusion
elif page == pages[5]:
    st.header("📌 Conclusion")
    st.write("**Conclusion**")



