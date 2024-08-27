import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    st.caption("""**Cursus** : Data Analyst | **Formation** : Formation Continue | **Mois** : Janvier 2024 """)
    st.caption("""**Groupe** : Christophe MONTORIOL, Issam YOUSR, Gwilherm DEVALLAN, Yacine OUDMINE""")
     # Ajouter l'image du bandeau
    st.image('https://raw.githubusercontent.com/ChristopheMontoriol/French_Industry_Janv24/main/data/Bandeau_FrenchIndustry.png', use_column_width=True)
    st.write("""
        L’objectif premier de ce projet est d’observer et de comprendre quelles sont les inégalités salariales en France. 
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

    st.subheader("Disparité salariale homme/femme")
    
    # Menu déroulant pour la disparité salariale
    disparite_options = ["Disparité salariale par catégorie socioprofessionnelle", "Disparité salariale par tranche d'âge"]
    disparite_choice = st.selectbox("Sélectionnez une visualisation pour la disparité salariale :", disparite_options)
    
    # Visualisation en fonction du choix de l'utilisateur pour la disparité salariale
    if disparite_choice == disparite_options[0]:
        # Disparité salariale par catégorie socioprofessionnelle
        categories = ['Cadres', 'Cadres moyens', 'Employés', 'Travailleurs']
        disparites = [17.60531468314386, 9.887706605652797, 2.472865187964315, 14.680015141858643]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(categories, disparites, color='skyblue')

        ax.set_title('Disparité salariale par catégorie socioprofessionnelle')
        ax.set_xlabel('Catégorie socioprofessionnelle')
        ax.set_ylabel('Disparité salariale (%)')

        plt.xticks(rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    elif disparite_choice == disparite_options[1]:
        # Disparité salariale par tranches d'âge
        tranches_age = ['18-25 ans', '26-50 ans', 'Plus de 50 ans']
        disparites_age = [4.286591078294969, 11.745237278240928, 20.02852196164705]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(tranches_age, disparites_age, color='lightgreen')
        ax.set_title('Disparité salariale par tranche d\'âge')
        ax.set_xlabel('Tranche d\'âge')
        ax.set_ylabel('Disparité salariale (%)')
        plt.xticks(rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    st.subheader("Comparaison de salaire homme/femme")

    # Menu déroulant pour la comparaison des salaires entre hommes et femmes
    comparaison_options = ["Comparaison par catégorie socioprofessionnelle", "Comparaison par tranche d'âge"]
    comparaison_choice = st.selectbox("Sélectionnez une visualisation pour la comparaison des salaires :", comparaison_options)
    
    # Visualisation en fonction du choix de l'utilisateur pour la comparaison des salaires
    if comparaison_choice == comparaison_options[0]:
        # Boîte à moustaches pour chaque catégorie socioprofessionnelle : Hommes et femmes 
        salaires_hommes = salaire[['salaire_cadre_homme', 'salaire_cadre_moyen_homme', 'salaire_employe_homme', 'salaire_travailleur_homme']]
        salaires_femmes = salaire[['salaire_cadre_femme', 'salaire_cadre_moyen_femme', 'salaire_employe_femme', 'salaire_travailleur_femme']]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Boîte à moustaches pour les salaires des hommes
        ax.boxplot([salaires_hommes[col] for col in salaires_hommes.columns], positions=[1, 2, 3, 4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="blue"))

        # Boîte à moustaches pour les salaires des femmes
        ax.boxplot([salaires_femmes[col] for col in salaires_femmes.columns], positions=[1.4, 2.4, 3.4, 4.4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="red"))

        ax.set_title('Comparaison des salaires entre hommes et femmes pour chaque catégorie socioprofessionnelle')
        ax.set_xlabel('Catégorie socioprofessionnelle')
        ax.set_ylabel('Salaire')
        plt.xticks([1.2, 2.2, 3.2, 4.2], ['Cadre', 'Cadre moyen', 'Employé', 'Travailleur'])
        ax.grid(True)
        st.pyplot(fig)

    elif comparaison_choice == comparaison_options[1]:
        # Boîte à moustaches pour chaque tranche d'âge : Hommes et femmes 
        salaires_hommes = salaire[['salaire_18-25_homme', 'salaire_26-50_homme', 'salaire_+50_homme']]
        salaires_femmes = salaire[['salaire_18-25_femme', 'salaire_26-50_femme', 'salaire_+50_femme']]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Boîte à moustaches pour les salaires des hommes
        ax.boxplot([salaires_hommes[col] for col in salaires_hommes.columns], positions=[1, 2, 3], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="blue"))

        # Boîte à moustaches pour les salaires des femmes
        ax.boxplot([salaires_femmes[col] for col in salaires_femmes.columns], positions=[1.4, 2.4, 3.4], widths=0.4, labels=salaires_hommes.columns, boxprops=dict(color="red"))

        ax.set_title("Comparaison des salaires entre hommes et femmes pour chaque tranche d'âge")
        ax.set_xlabel("Tranche d'âge")
        ax.set_ylabel('Salaire')
        plt.xticks([1.2, 2.2, 3.2], ['18-25 ans', '26-50 ans', 'Plus de 50 ans'])
        ax.grid(True)
        st.pyplot(fig)


# Page de Modélisation
elif page == pages[3]:
    st.header("🧩 Modélisation")
    st.subheader("Objectif")
    st.write("Prédire le salaire net moyen en fonction des features.")
    
    if st.button("Modèles étudiés") :
        st.subheader("Liste des modèles")
        st.markdown("""
                    Afin de déterminer le plus performant possible, nous avons étudié plusieurs modèles de machine learning:
                    - Régression linéaire
                    - Forêt aléatoire
                    - Clustering
        """)


        st.subheader("Exécution des modèles")
        st.markdown("""
                    Pour chaque modèle appliqué, nous avons suivi les étapes suivantes :
                    1. Instanciation du modèle.
                    2. Entrainement du modèle sur l'ensemble du jeu d'entraînement X_train et y_train.
                    3. Prédictions sur l'ensemble du jeu de test X_test et y_test.
                    4. Evaluation de la performance des modèles en utilisant les métriques appropriées.
                    5. Interprétation des coefficients pour comprendre l'impact de chaque caractéristique sur la variable cible.
                    6. Optimisation du modèle : variation des paramètres, sélection des features utilisées, discrétisation des valeurs.
                    7. Visualisation et analyse des résultats.
                """)
    if st.button("Modèle retenu") :
        data = {
        'Modèles': ['Forêt aléatoire sans optimisation', 'Forêt aléatoire avec optimisation',  'Forêt aléatoire avec ratio H/F','Forêt aléatoire avec discrétisation','Régression linéaire 1','Régression linéaire 2'],
        'R² train': [0.9994,0.9441,0.9491,0.9456,0.9993,0.9946],
        'R² test': [0.9977,0.8892,0.9376,0.9140,0.9996,0.9938],
        'MSE test': [0.0117, 0.5903,0.3755,0.4577,0.0022,0.0344],
        'MAE test': [0.0747,0.5250,0.4523,0.5240,0.0377,0.1319],
        'RMSE test': [0.1084,0.7683, 0.6127,0.6765,0.0474,0.1855]
        }

        
        # Création du DataFrame
        tab = pd.DataFrame(data)
        tab.index = tab.index #+ 1
        # Trouver l'index de la ligne correspondant à "Forêt aléatoire avec discrétisation"
        rf_index = tab[tab['Modèles'] == 'Forêt aléatoire avec discrétisation'].index

        # Appliquer un style personnalisé à la ligne spécifique
        styled_tab = tab.style.apply(lambda x: ['background: #27dce0' if x.name in rf_index else '' for i in x], axis=1)


        # Afficher le tableau avec le style appliqué
        st.subheader("Synthèse des métriques de performance")
        st.table(styled_tab)
        st.markdown("""
                        ##### Choix du modèle :
                        - Les modèles de régression linaires 1 & 2 font de l'overfitting même après optimisation.                     
                        Ils sont donc disqualifiés.
                        - Critères de choix pour le modèle Forêt aléatoire :                    
                        Les R² ne montrent pas d'overfitting et sont proches de 0.9.                                
                        Les erreurs restent acceptables.
                        """)
        st.write("")
        st.write("#### Modèle retenue : Forêt aléatoire avec discrétisation.")


    if st.button("Evaluation graphique du modèle") :
 
        st.subheader("Dispersion des résidus & distributions des résidus")
        image_qqplot = "https://zupimages.net/up/24/35/r6ed.png"
        st.image(image_qqplot, use_column_width=True)

        st.write("")
        st.write("")

        st.subheader("Comparaison des predictions VS réelles & QQ plot des résidus")
        image_cumul_residus = "https://zupimages.net/up/24/35/t9c6.png"
        st.image(image_cumul_residus, use_column_width=True)
        
        st.markdown("""
                    ##### Conclusions :         
                    - Distributions relativement centrées autour de zéro
                    - Distribution normale des résidus
                    - Très peu de points au dela de +/-2
                    - Les résultats obtenus sont plutot uniformes pour toute la plage des données
                    """)
        
        st.write("")
        st.write("")



# Page de Prédiction
elif page == pages[4]:
    st.header("🔮 Prédiction")
    st.subheader('Simulation de Prédiction avec Random Forest Regressor')

# Page de Conclusion
elif page == pages[5]:
    st.header("📌 Conclusion")
    st.write("**Conclusion**")



