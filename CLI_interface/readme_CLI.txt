Fichiers créés
cli_interface.py — Interface CLI avec :
Classification de sentiment : utilise fine_tuned_sentiment_model pour classifier les reviews (positif/négatif)
Réponses contextuelles : utilise embedding_model pour rechercher dans le corpus et generator_model pour générer des réponses
Interface interactive avec commandes simples

requirements.txt — Dépendances nécessaires

Utilisation
Pour utiliser l'interface CLI :

Installer les dépendances (si nécessaire) :
   pip install -r requirements.txt

Lancer l'interface :
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