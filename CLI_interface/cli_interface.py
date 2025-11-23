#!/usr/bin/env python3
"""
Interface CLI pour classifier des reviews de films et répondre à des questions sur les films.
Utilise:
- fine_tuned_sentiment_model: pour classifier le sentiment des reviews
- embedding_model: pour créer des embeddings et rechercher dans le corpus
- generator_model: pour générer des réponses contextuelles
"""

import os
import sys
import numpy as np

# Gestion des erreurs de chargement de PyTorch
try:
    import torch
except OSError as e:
    print("="*60)
    print("ERREUR: Impossible de charger PyTorch")
    print("="*60)
    print(f"Erreur: {e}")
    print("\nCe problème est souvent causé par:")
    print("1. Incompatibilité entre Python 3.13 et PyTorch")
    print("2. DLL manquantes (Visual C++ Redistributables)")
    print("3. Installation corrompue de PyTorch")
    print("\nSolutions:")
    print("1. Exécutez: python fix_pytorch_issue.py")
    print("2. Ou réinstallez PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    print("3. Ou utilisez Python 3.11 ou 3.12 (recommandé)")
    print("="*60)
    sys.exit(1)
except ImportError as e:
    print("="*60)
    print("ERREUR: PyTorch n'est pas installé")
    print("="*60)
    print("Installez PyTorch avec:")
    print("pip install torch torchvision torchaudio")
    print("="*60)
    sys.exit(1)

from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
import warnings
warnings.filterwarnings('ignore')


class FilmReviewCLI:
    def __init__(self):
        """Initialise tous les modèles nécessaires."""
        print("Chargement des modèles...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")
        
        # Charger le modèle d'embedding
        print("Chargement du modèle d'embedding...")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('embedding_model')
        self.embedding_model_base = AutoModel.from_pretrained('embedding_model').to(self.device)
        self.embedding_model_base.eval()
        
        # Essayer d'utiliser SentenceTransformer si disponible
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer('embedding_model', device=self.device)
                self.use_sentence_transformer = True
            except:
                self.use_sentence_transformer = False
        else:
            self.use_sentence_transformer = False
        
        # Charger le modèle de sentiment
        print("Chargement du modèle de sentiment...")
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2
            )
            self.sentiment_model = PeftModel.from_pretrained(base_model, 'fine_tuned_sentiment_model')
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained('fine_tuned_sentiment_model')
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de sentiment: {e}")
            raise
        
        # Charger le modèle de génération (T5 - encoder-decoder)
        print("Chargement du modèle de génération...")
        try:
            self.generator_model = AutoModelForSeq2SeqLM.from_pretrained('generator_model')
            self.generator_model.to(self.device)
            self.generator_model.eval()
            self.generator_tokenizer = AutoTokenizer.from_pretrained('generator_model')
            # S'assurer qu'il y a un pad_token
            if self.generator_tokenizer.pad_token is None:
                self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        except Exception as e:
            print(f"Erreur lors du chargement du modèle de génération: {e}")
            raise
        
        # Charger le corpus
        print("Chargement du corpus...")
        self.corpus_texts = np.load('corpus_texts.npy', allow_pickle=True)
        corpus_embeddings_np = np.load('corpus_embeddings.npy', allow_pickle=True)
        
        # Convertir en liste si nécessaire
        if isinstance(self.corpus_texts, np.ndarray):
            if self.corpus_texts.dtype == object:
                self.corpus_texts = self.corpus_texts.tolist()
            else:
                self.corpus_texts = [str(text) for text in self.corpus_texts]
        
        # Convertir les embeddings en tensor
        if isinstance(corpus_embeddings_np, np.ndarray):
            self.corpus_embeddings = torch.tensor(corpus_embeddings_np, dtype=torch.float32).to(self.device)
        else:
            self.corpus_embeddings = torch.tensor(corpus_embeddings_np, dtype=torch.float32).to(self.device)
        
        print(f"Corpus chargé: {len(self.corpus_texts)} textes, embeddings shape: {self.corpus_embeddings.shape}")
        
        print("Tous les modèles sont chargés!\n")
    
    def get_embedding(self, text):
        """Crée un embedding pour un texte donné."""
        try:
            if self.use_sentence_transformer:
                return self.embedding_model.encode(text, convert_to_tensor=True, device=self.device)
            else:
                # Utiliser le modèle DistilBERT directement
                inputs = self.embedding_tokenizer(text, return_tensors='pt', truncation=True, 
                                                 padding=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.embedding_model_base(**inputs)
                    # Utiliser le [CLS] token
                    embedding = outputs.last_hidden_state[:, 0, :]
                    # Normaliser
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                return embedding.squeeze(0)
        except Exception as e:
            print(f"Erreur lors de la création de l'embedding: {e}")
            return None
    
    def classify_sentiment(self, review):
        """Classifie le sentiment d'une review (positif/négatif)."""
        inputs = self.sentiment_tokenizer(
            review, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][predicted_class].item()
        
        sentiment = "positif" if predicted_class == 1 else "négatif"
        return sentiment, confidence
    
    def search_corpus(self, query, top_k=3):
        """Recherche les textes les plus pertinents dans le corpus."""
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        # S'assurer que query_embedding est 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # S'assurer que les dimensions correspondent
        if query_embedding.shape[1] != self.corpus_embeddings.shape[1]:
            # Si les dimensions ne correspondent pas, on prend les premières dimensions
            min_dim = min(query_embedding.shape[1], self.corpus_embeddings.shape[1])
            query_embedding = query_embedding[:, :min_dim]
            corpus_embeddings = self.corpus_embeddings[:, :min_dim]
        else:
            corpus_embeddings = self.corpus_embeddings
        
        # Calculer les similarités cosinus
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding, 
            corpus_embeddings, 
            dim=1
        )
        
        # Obtenir les top_k indices
        k = min(top_k, len(self.corpus_texts))
        top_indices = torch.topk(similarities, k).indices.cpu().tolist()
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.corpus_texts[idx],
                'similarity': similarities[idx].item()
            })
        
        return results
    
    def generate_response(self, query, context_texts, max_length=150):
        """Génère une réponse contextuelle basée sur la requête et le contexte."""
        # Construire le prompt avec le contexte
        context = "\n".join([f"- {text['text'][:200]}..." if len(text['text']) > 200 else f"- {text['text']}" 
                            for text in context_texts[:2]])
        prompt = f"Contexte sur les films:\n{context}\n\nQuestion: {query}\nRéponse:"
        
        # Pour T5 (encoder-decoder), encoder le prompt comme input
        inputs = self.generator_tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512  # Longueur max pour l'encoder
        ).to(self.device)
        
        with torch.no_grad():
            # Pour les modèles seq2seq comme T5, la génération prend seulement max_length pour la sortie
            outputs = self.generator_model.generate(
                inputs.input_ids,
                max_length=max_length,  # Longueur de la réponse générée
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator_tokenizer.pad_token_id,
                eos_token_id=self.generator_tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Pour T5, la sortie est directement la réponse générée (sans le prompt)
        response = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        response = response.split('\n')[0]  # Prendre la première ligne
        if not response:
            response = "Je n'ai pas pu générer de réponse pertinente."
        
        return response
    
    def answer_question(self, query):
        """Répond à une question sur les films en utilisant le corpus et le générateur."""
        print(f"\nRecherche dans le corpus pour: '{query}'...")
        context_texts = self.search_corpus(query, top_k=3)
        
        if not context_texts:
            return "Désolé, je n'ai pas trouvé d'information pertinente dans le corpus."
        
        print(f"Contexte trouvé ({len(context_texts)} résultats)")
        print("Génération de la réponse...")
        
        response = self.generate_response(query, context_texts)
        return response
    
    def run(self):
        """Lance l'interface CLI interactive."""
        print("=" * 60)
        print("Interface CLI - Classification et Questions sur les Films")
        print("=" * 60)
        print("\nCommandes disponibles:")
        print("  1. 'classify <review>' - Classifier le sentiment d'une review")
        print("  2. 'ask <question>' - Poser une question sur les films")
        print("  3. 'quit' ou 'exit' - Quitter l'application")
        print("\n" + "=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Au revoir!")
                    break
                
                if user_input.startswith('classify '):
                    review = user_input[9:].strip()
                    if review:
                        sentiment, confidence = self.classify_sentiment(review)
                        print(f"\nSentiment: {sentiment.upper()}")
                        print(f"Confiance: {confidence:.2%}\n")
                    else:
                        print("Veuillez fournir une review à classifier.\n")
                
                elif user_input.startswith('ask '):
                    question = user_input[4:].strip()
                    if question:
                        response = self.answer_question(question)
                        print(f"\nRéponse: {response}\n")
                    else:
                        print("Veuillez poser une question.\n")
                
                else:
                    print("Commande non reconnue. Utilisez 'classify <review>' ou 'ask <question>'.\n")
            
            except KeyboardInterrupt:
                print("\n\nAu revoir!")
                break
            except Exception as e:
                print(f"\nErreur: {e}\n")


def main():
    """Point d'entrée principal."""
    try:
        cli = FilmReviewCLI()
        cli.run()
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

