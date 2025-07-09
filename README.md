# Analyse de Segmentation des Magasins

Ce projet est une application Streamlit pour l'analyse, la segmentation et la recommandation stratégique des magasins à partir de données Excel.

## Fonctionnalités
- Exploration et visualisation des données
- Prétraitement et standardisation
- Détermination du nombre optimal de clusters (K-Means)
- Clustering et analyse des segments
- Visualisation PCA
- Génération de rapport
- Quiz interactifs

## Démarrage rapide

1. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```
2. **Lancer l'application**
   ```bash
   streamlit run app.py
   ```
3. **Charger vos données**
   - Utilisez le panneau latéral pour uploader un fichier Excel au format attendu.

## Déploiement sur Streamlit Cloud
1. Poussez ce projet sur un dépôt GitHub public.
2. Rendez-vous sur [https://share.streamlit.io/](https://share.streamlit.io/) et connectez votre compte GitHub.
3. Sélectionnez ce dépôt et le fichier `app.py` comme point d'entrée.
4. Cliquez sur "Deploy".

## Format attendu des données
Le fichier Excel doit contenir les colonnes suivantes :
- magasin
- chiffre d'affaires (dh)
- clients/jour
- surface (m²)
- employés
- zone

## Auteur
Développé par Mohamed Saidi pour l'analyse stratégique du réseau de magasins. 