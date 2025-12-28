## Présentation
Bienvenue sur le projet fantastique et ambitieux de Mohamed et Ken !
Ce projet a pour objectif d’aider les **livreurs** à organiser leur tournée.  
Un livreur doit livrer de nombreux colis chaque jour à des adresses différentes. L’enjeu est donc de trouver le meilleur ordre de livraison afin de réduire le temps de trajet.

Ce problème est connu sous le nom de **problème du voyageur de commerce**. Le problème du voyageur de commerce consiste à trouver le meilleur ordre pour visiter plusieurs adresses, afin de parcourir la distance la plus courte possible. Quand le nombre d’adresses augmente, le problème devient plus complexe et nécessite une solution informatique.
Dans ce projet, ce problème est appliqué au cas des **livreurs**, afin d’optimiser leurs tournées de livraison.

Notre objectif est de proposer une solution simple afin de déterminer l’itinéraire le plus court possible à partir d’une liste d’adresses.

## Principe du projet
1. Génération et préparation de notre propre dataset : les données sont extraites, organisées et transformées à partir de différentes adresses afin de créer un jeu de données exploitable pour le machine learning.
2. Entraînement d’un modèle de machine learning basé sur l'XGBoost : le modèle apprend à prédire l’ordre optimal des adresses afin de minimiser le temps de trajet.
3. Il teste les performances du modèle
4. Il affiche les résultats

Tout le processus est **automatique** et se lance avec un seul fichier.

---

## Structure du projet

- `main.py`  
  Fichier principal à lancer. Il exécute toutes les étapes du projet.

- `data.py`  
  Préparation des données utilisées par le modèle.

- `model.py`  
  Entraînement du modèle de machine learning.

- `model_2.py`  
  Deuxième méthode d’entraînement pour comparer les résultats.

- `test.py`  
  Test des performances du modèle.

- `utils.py`  
  Fonctions utiles pour le fonctionnement du projet.

