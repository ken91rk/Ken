Bienvenue sur le projet fantastique et ambitieux de Mohamed et Ken,
Savez-vous que 1,7 milliard de colis sont livrés par an en France. Mais comment un système aussi complexe fonctionne ? C'est grâce à un réseaux de livreurs organiser dans toute la France, séparer en commune généralement. Un livreur livre en moyenne 140 colis par jours à des adresses différentes. Mais comment établir sa tournée pour qu'elle dure le moins de temps. Ce problème se nomme " Le problème du voyageur". 

Notre objectif :  Trouver le meilleur ordre d'adresse pour prendre le moins de temps possible pour des adresse dans Paris. 

En X :  le temps de trajet 
En Y : l'ordre des adresse


On decompose notre adresse en 3 etapes : 
- Créer notre Dataseet : 
          - générer notre dataseet
          - Analyser nos donées ( distance en fonction du temps, histogramme : avec plage horaire ( 0,2, 0,5; 0,10)
- Entraîner notre LLM
- Trouver l'ordre optimal

1) Création de notre Dataseet

Notre objectif est de créer un dataseet. Pour notre dataseet on établit : 
En X :  la distance
En Y : l'ordre des adresse


