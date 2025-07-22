Explication détaillée par étapes de fichier main.py dans chatbot-nlp-anayzer:
1. Chargement des données JSON (load_conversations)
Parcourt un dossier contenant plusieurs fichiers JSON (chacun représentant une session de conversation).

Charge toutes les conversations de chaque fichier.

Ajoute un champ session_id à chaque échange (extrait du nom du fichier).

Combine tous les échanges dans un seul DataFrame pandas.

2. Nettoyage du texte (clean_text + preprocess_text)
Pour chaque message utilisateur (user_input) et réponse bot (bot_response):

Met le texte en minuscules.

Supprime les espaces multiples.

Enlève les caractères non alphabétiques sauf la ponctuation utile (?.!,).

Supprime les espaces inutiles en début/fin.

Ajoute ces textes nettoyés en nouvelles colonnes user_input_clean et bot_response_clean.

3. Sauvegarde des conversations nettoyées (save_cleaned_conversations)
Regroupe les échanges par session.

Sauvegarde chaque session nettoyée dans un fichier JSON distinct dans un dossier de sortie.

Permet de garder une trace des données nettoyées, exploitables plus facilement.

4. Détection des questions (detect_questions)
Pour chaque message utilisateur nettoyé, détermine si c’est une question.

Critères simples : présence de ? ou d’un mot interrogatif courant (quoi, comment, pourquoi, etc.).

Ajoute une colonne booléenne is_question.

5. Calcul des écarts de temps entre messages (add_time_differences)
Convertit la colonne timestamp en datetime.

Trie les échanges par session et date.

Calcule la différence en secondes entre deux messages consécutifs dans une même session.

Ajoute la colonne time_diff_sec.

6. Détection des hésitations (detect_hesitations)
Recherche dans les messages utilisateur nettoyés des expressions typiques d’hésitation (euh, mmh, hmm, ...).

Ajoute une colonne booléenne has_hesitation.

7. Analyse des sentiments (analyze_sentiment)
Utilise un modèle pré-entraîné de Transformers (nlptown/bert-base-multilingual-uncased-sentiment) pour analyser le sentiment des messages utilisateur.

Pour chaque message, récupère :

Le label de sentiment (ex : 1 star à 5 stars).

La confiance (score).

Ajoute ces infos dans les colonnes sentiment_label et sentiment_score.

8. Longueur des messages (add_text_lengths)
Compte le nombre de mots dans chaque message utilisateur et chaque réponse bot.

Ajoute ces valeurs dans user_input_len et bot_response_len.

9. Détection des questions répétées (detect_repeated_questions)
Compare chaque question utilisateur avec les suivantes dans la session.

Calcule la similarité textuelle (avec rapidfuzz.fuzz.ratio).

Si la similarité dépasse un seuil (90%), marque la question comme répétée.

Ajoute une colonne booléenne is_repeated_question.

10. Évaluation du niveau de compréhension par session (evaluate_comprehension)
Se base sur les questions posées et le taux de répétition.

Calcule par session :

Nombre total de questions.

Nombre de questions répétées.

Taux de répétition = répétées / total.

Attribue un score de compréhension selon ce taux :

< 20% → bon (1)

entre 20% et 50% → moyen (0.5)

50% → faible (0)

Retourne un DataFrame avec ces scores.

11. Agrégation des features par session (aggregate_features)
Regroupe par session plusieurs mesures moyennes ou totales :

Score moyen de sentiment.

Délai moyen entre messages.

Nombre total d’échanges.

Taux moyen d’hésitation.

Nombre total de questions.

Nombre moyen de mots par message.

Score moyen de compréhension.

Renomme les colonnes avec des noms français explicites.

12. Étiquetage de la tonalité globale (sentiment_label)
Traduit le score moyen de sentiment en tonalité globale qualitative :

Score ≥ 3.5 → "positif"

Score ≤ 2.5 → "négatif"

Sinon → "neutre"

13. Détection des sessions difficiles (detect_sessions_difficiles)
Marque une session comme difficile si elle répond à ces critères :

Tonalité négative.

Taux d’hésitation > 30%.

Score de compréhension faible (< 0.5).

Ajoute une colonne booléenne session_difficile.

14. Programme principal (main)
Définition des chemins d’entrée et de sortie.

Appelle toutes les fonctions dans l’ordre :

Chargement → Nettoyage → Sauvegarde → Détection questions → Calcul temps → Hésitations → Sentiments → Longueurs → Questions répétées → Évaluation compréhension → Agrégation → Tonalité → Détection sessions difficiles.

Sauvegarde les résultats finaux dans des fichiers CSV (comprehension_scores.csv, features_sessions.csv).

Affiche un aperçu des résultats.

En résumé
Le script traite automatiquement des données brutes de conversations, en extrayant des indicateurs clés (questions, sentiments, hésitations, répétitions), calcule des scores par session, et produit un résumé exploitable pour analyser la qualité ou difficulté des échanges entre utilisateur et chatbot.
** la différence entre les deux fichiers CSV :
 1. comprehension_scores.csv
Contient les scores liés à la compréhension calculés uniquement à partir des questions posées par les utilisateurs.

C’est un tableau résumé par session qui donne :

Le nombre total de questions posées (total_questions).

Le nombre de questions répétées (repeated_questions).

Le taux de répétition (repetition_rate).

Le score de compréhension associé (comprehension_score), basé sur ce taux de répétition.

Ce fichier est utile pour comprendre à quel point les utilisateurs comprennent les réponses du chatbot, en regardant combien ils posent de questions répétées.

2. features_sessions.csv
Contient une agrégation plus complète des indicateurs par session, qui inclut :

Le score de sentiment moyen des messages utilisateur.

Le délai moyen entre messages.

Le nombre total d’échanges (messages).

Le taux moyen d’hésitation.

Le nombre total de questions.

La longueur moyenne des messages utilisateur.

Le score moyen de compréhension (fusionné depuis le fichier précédent).

Une étiquette de tonalité globale (positif, neutre, négatif).

Une indication si la session est considérée comme difficile (session_difficile).

Ce fichier donne donc un profil global et riche de chaque session, combinant plusieurs aspects comportementaux et linguistiques.


