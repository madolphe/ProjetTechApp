# ProjetTechApp

Le projet suivant est un exercice proposé par le cours IFT712 "Technique d'apprentissage". L'objectif est d'implémenter 6 outils de classification sur un jeu de données "réelles". Le corpus utilisé se retrouve à l'adresse : https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia. 

# Installation

Requis: Python 3.5+ | Linux, Mac OS X, Windows

```sh
pip install pipenv
```
Puis dans le dossier du projet:  

```sh
pipenv install 
```
Le pipfile permettra l'installation de toutes les dépendances nécessaires à l'utilisation du projet. 
Puis pour exécuter des commandes dans cet environnement virtuel: 

```sh
pipenv shell
```
Note: les images ne sont pas présentes sur ce repo, cependant des fichiers .npy contenant les caractéristiques extraites sont présents dans le dossier Data/chest_xray_feature_vect.

# Préparez-vous :

Pour lancer le notebook principal se lance:
```sh
pipenv run jupyter notebook
```
Puis sélectionner bench-test.ipynb !

# Description

Ce dossier contient l'implémentation de 6 outils de classification testés sur des images de thorax présentant (ou non) une anomalie (pneumonie). L'ensemble du travail est résumé dans le notebook bench-test.ipynb. La figure suivante présente un exemple de métriques utilisé pour la comparaison des méthodes. 
<p float="left">
<img src="https://github.com/madolphe/ProjetTechApp/blob/master/Rapport-PDF/images/ROC.png" width=400 height="400"/>
<img src="https://github.com/madolphe/ProjetTechApp/blob/master/Rapport-PDF/images/precision_rappel.png" width=400 height="400"/>
<p/>
