================================================================================
                    README - HACKATHON 2 : LLM FINEMENT AJUSTÉ
          Classification de critiques de film (positive/négative) + Réponses Contextuelles (RAG)
================================================================================

PROJET
------
Ce projet implémente un système de traitement du langage naturel combinant :
1. Un modèle de classification de sentiments (positif/négatif) fine-tuné avec LoRA
2. Un système RAG (Retrieval-Augmented Generation) pour générer des réponses
   contextuelles basées sur des critiques de films

FICHIER PRINCIPAL
-----------------
- Hackathon2_Groupe.ipynb : Notebook Jupyter contenant l'implémentation complète

OBJECTIFS
---------
* Classification binaire de critiques de films (positive ou négative)
* Génération de réponses contextuelles basées sur un corpus de critiques (RAG)

ARCHITECTURE DU SYSTÈME
------------------------

1. CLASSIFICATION DE SENTIMENTS
   - Modèle de base : DistilBERT (distilbert-base-uncased)
   - Technique : Fine-tuning avec LoRA (Low-Rank Adaptation)
   - Type : Classification binaire
   - Dataset : IMDB Movie Reviews
     * Entraînement : 10 000 échantillons (5 000 positives, 5 000 négatives)
     * Test : 2 000 échantillons (1 000 positives, 1 000 négatives)
   - Performance : 90.45% de précision, perte de validation : 0.2558

2. SYSTÈME RAG (RETRIEVAL-AUGMENTED GENERATION)
   - Embeddings : Sentence-BERT (sentence-transformers/all-MiniLM-L6-v2)
   - Recherche : k-Nearest Neighbors (kNN) avec similarité cosinus
   - Génération : Flan-T5 (google/flan-t5-base)
   - Corpus : 10 000 critiques de films IMDB

MODÈLES ET PARAMÈTRES CLÉS
---------------------------

1. CLASSIFICATION - DistilBERT + LoRA
   Modèle : distilbert-base-uncased
   Paramètres LoRA :
   - r = 8 (rank)
   - lora_alpha = 16
   - lora_dropout = 0.1
   - target_modules = ["q_lin", "v_lin"]
   Paramètres d'entraînement :
   - num_train_epochs = 2
   - per_device_train_batch_size = 8
   - per_device_eval_batch_size = 8
   - learning_rate = 2e-4
   - weight_decay = 0.01
   - max_length = 512
   - fp16 = True
   Résultat : 739 586 paramètres entraînables (1.09% du modèle total)

2. EMBEDDINGS - Sentence-BERT
   Modèle : sentence-transformers/all-MiniLM-L6-v2
   Paramètres :
   - Dimension des embeddings : 384
   - normalize_embeddings = True (normalisation L2)
   - batch_size = 32
   - Device : CUDA si disponible

3. RECHERCHE - kNN
   Paramètres :
   - n_neighbors = 10
   - metric = "cosine"
   - top_k pour retrieval : 5 (par défaut)

4. GÉNÉRATION - Flan-T5
   Modèle : google/flan-t5-base
   Paramètres de génération :
   - max_new_tokens = 150
   - min_length = 30
   - temperature = 0.7
   - do_sample = True
   - no_repeat_ngram_size = 2
   - max_length (prompt) = 512

ÉTAPES D'IMPLÉMENTATION
------------------------

ÉTAPE 1 : Installation et Importation
- Installation des librairies : torch, transformers, datasets, accelerate,
  peft, scikit-learn, numpy, pandas, tqdm, sentence-transformers
- Importation des modules nécessaires

ÉTAPE 2 : Chargement et Préparation des Données
- Chargement du dataset IMDB depuis Hugging Face
- Échantillonnage équilibré (5 000 positives / 5 000 négatives pour train,
  1 000 / 1 000 pour test)
- Tokenization avec DistilBERT tokenizer (max_length=512, truncation=True,
  padding=True)

ÉTAPE 3 : Fine-tuning du Modèle de Classification
- Configuration LoRA (r=8, lora_alpha=16, lora_dropout=0.1)
- Entraînement sur 2 époques
- Évaluation et sauvegarde du modèle

ÉTAPE 4 : Construction du Système RAG
- Création d'embeddings pour le corpus (10 000 critiques)
- Construction de l'index kNN pour la recherche de similarité
- Implémentation de la fonction retrieve_documents()
- Configuration du modèle de génération Flan-T5
- Implémentation de la fonction generate_response()

ÉTAPE 5 : Évaluation et Tests
- Évaluation sur 50 requêtes de test
- Tests de robustesse (requêtes vagues, contradictoires, non liées)
- Métriques : longueur de réponse, répétitions, présence de mots-clés, latence

PRÉREQUIS
---------
- Python 3.x
- GPU recommandé (CUDA) pour accélérer l'entraînement et l'inférence
- Google Colab (recommandé) ou environnement local avec GPU
- Connexion Internet pour télécharger les modèles et datasets

INSTALLATION
------------
Les dépendances sont installées dans le notebook via :
!pip install -q torch transformers datasets accelerate peft scikit-learn
numpy pandas tqdm openpyxl sentence-transformers

FICHIERS GÉNÉRÉS
----------------
- fine_tuned_sentiment_model/ : Modèle de classification fine-tuné
- embedding_model/ : Modèle d'embeddings sauvegardé
- generator_model/ : Modèle de génération sauvegardé
- corpus_texts.npy : Corpus de textes (10 000 critiques)
- corpus_embeddings.npy : Embeddings du corpus (shape: 10000, 384)
- evaluation_results_optimized.csv : Résultats d'évaluation
- results/ : Dossier de checkpoints d'entraînement
- logs/ : Logs d'entraînement

UTILISATION
-----------
1. Ouvrir le notebook Hackathon2_Groupe.ipynb dans Google Colab ou Jupyter
2. Activer le GPU (Runtime → Change runtime type → GPU)
3. Exécuter les cellules dans l'ordre
4. Pour tester la classification :
   sentiment_pipeline("This movie was absolutely fantastic!")
5. Pour tester le système RAG :
   query = "What do people think about the special effects?"
   retrieved = retrieve_documents(query, top_k=5)
   response = generate_response(query, retrieved)

RÉSULTATS ET PERFORMANCES
--------------------------
- Précision de classification : 90.45% sur le jeu de validation
- Perte de validation : 0.2558
- Paramètres entraînables : 739 586 (1.09% du modèle total)
- Latence moyenne : ~1.1 secondes par requête (avec GPU)
- Temps d'entraînement : ~10-15 minutes (avec GPU)

OPTIMISATIONS IMPLÉMENTÉES
---------------------------
1. Fine-tuning LoRA : Réduction des paramètres entraînables à 1.09%
2. Normalisation L2 des embeddings : Amélioration de la recherche de similarité
3. Nettoyage de texte : Suppression des balises HTML et normalisation
4. Prompt engineering : Optimisation des prompts pour Flan-T5
5. Utilisation de Sentence-BERT : Embeddings optimisés pour la similarité
6. Précision mixte (fp16) : Accélération de l'entraînement

TESTS DE ROBUSTESSE
-------------------
Le système a été testé sur :
- Requêtes vagues ("What about it?")
- Requêtes contradictoires ("This movie is both amazing and terrible")
- Requêtes non liées ("What is the weather like today?")

OBSERVATIONS
------------
- Le système gère bien les requêtes normales sur les films
- Les requêtes vagues peuvent produire des réponses moins pertinentes
- Les requêtes contradictoires sont résolues en choisissant un sentiment dominant
- Les requêtes non liées peuvent retourner des documents faiblement similaires

AMÉLIORATIONS FUTURES
---------------------
- Augmenter la taille du dataset d'entraînement
- Améliorer l'ingénierie des prompts
- Fine-tuner le générateur sur des critiques de films
- Utiliser des techniques de retrieval plus sophistiquées
  (ex: dense passage retrieval)
- Implémenter une meilleure détection et prévention des répétitions
- Créer une interface Streamlit pour une meilleure expérience utilisateur

NOTES TECHNIQUES
----------------
- Le notebook est conçu pour fonctionner sur Google Colab avec GPU
- Les modèles sont automatiquement téléchargés depuis Hugging Face
- Les embeddings sont normalisés pour optimiser la recherche cosinus
- Le système utilise la précision mixte (fp16) pour accélérer l'entraînement

AUTEURS
-------
Équipe : RFC Flow

CADRE
-----
Hackathon 2 - Bootcamp Gen AI & Machine Learning
Septembre 2025

================================================================================
                            FIN DU README
================================================================================
