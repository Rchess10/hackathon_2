Fichiers créés
1) cli_interface.py — Interface CLI avec :
* Classification de sentiment : utilise fine_tuned_sentiment_model pour classifier les reviews (positif/négatif)
* Réponses contextuelles : utilise embedding_model pour rechercher dans le corpus et generator_model pour générer des réponses
=>Interface interactive avec commandes simples

2) ->lien googleDrive vers un dossier contenant tous les dossiers/fichiers A TELECHARGER AVANT DE LANCER cli_interface.py pour générer l'interface CLI:
	* le dossier 'fine_tuned_sentiment_model' contient les fichiers liés au modèle de classification binaire des critiques de film
	* le fichier 'corpus_texts.npy' contient les documents de références
	* le fichier 'corpus_embeddings.npy' contient les documents de référence numérisés pour la génération de réponse contextuelle
	* le dossier 'embedding_model' contient les fichiers liés à l'encodage des requêtes utilisateurs et des corpus de référence
	* le dossier 'generator_model' peremt de générer une réponse contextuelle en fonction de la requête utilisateur
	
3) requirements.txt — Dépendances nécessaires

=========
Utilisation
=========
Pour utiliser l'interface CLI :

*télécharger l'ensembles des fichiers/dossiers donnés sur le lien googleDrive: https://drive.google.com/drive/folders/1MmrUkkCmZOp8d4UXWv67G1cll8Cm9Iqn?usp=sharing

* Installer les dépendances (si nécessaire) :
   pip install -r requirements.txt

*Lancer l'interface :
   python cli_interface.py

Commandes disponibles :
classify <review> — Classifier le sentiment d'une review
Exemple : classify This movie was amazing!

ask <question> — Poser une question sur les films
Exemple : ask What are the best action movies?

quit ou exit — Quitter l'application


Fonctionnalités
Chargement automatique des modèles au démarrage
Détection automatique du device (CPU/GPU)
Recherche sémantique dans le corpus avec similarité cosinus
Génération de réponses contextuelles basées sur le corpus
Gestion d'erreurs
Le script est prêt à être utilisé. Il charge tous les modèles depuis les dossiers du répertoire courant et permet d'interagir via une interface CLI simple.