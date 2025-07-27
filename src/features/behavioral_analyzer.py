import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BehavioralFeatures:
    """Container for behavioral analysis features"""
    session_id: str
    hesitation_rate: float
    question_complexity: float
    engagement_score: float
    interaction_patterns: Dict[str, float]
    semantic_clusters: List[int]
    learning_pace: float
    confusion_indicators: float

class BehavioralAnalyzer:
    """Advanced behavioral analysis using NLP and ML techniques"""

    def __init__(self, embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embeddings_model = SentenceTransformer(embeddings_model)
        self.hesitation_patterns = [
            "euh", "mmh", "hmm", "ben", "alors", "donc",
            "comment dire", "je sais pas", "peut-être", "je pense que",
            "j'hésite", "c'est compliqué", "difficile à dire"
        ]

    def analyze_session_behavior(self, session_data: pd.DataFrame) -> BehavioralFeatures:
        """Comprehensive behavioral analysis for a single session"""
        session_id = session_data['session_id'].iloc[0]

        # Calculate various behavioral metrics
        hesitation_rate = self._calculate_hesitation_rate(session_data)
        question_complexity = self._analyze_question_complexity(session_data)
        engagement_score = self._calculate_engagement_score(session_data)
        interaction_patterns = self._analyze_interaction_patterns(session_data)
        semantic_clusters = self._perform_semantic_clustering(session_data)
        learning_pace = self._calculate_learning_pace(session_data)
        confusion_indicators = self._detect_confusion_indicators(session_data)

        return BehavioralFeatures(
            session_id=session_id,
            hesitation_rate=hesitation_rate,
            question_complexity=question_complexity,
            engagement_score=engagement_score,
            interaction_patterns=interaction_patterns,
            semantic_clusters=semantic_clusters,
            learning_pace=learning_pace,
            confusion_indicators=confusion_indicators
        )

    def _calculate_hesitation_rate(self, session_data: pd.DataFrame) -> float:
        """Calculate hesitation rate using advanced pattern matching"""
        total_messages = len(session_data)
        if total_messages == 0:
            return 0.0

        hesitation_count = 0
        for message in session_data['user_input']:
            message_lower = message.lower()

            # Direct pattern matching
            direct_hesitations = sum(1 for pattern in self.hesitation_patterns if pattern in message_lower)

            # Repetitive word patterns (e.g., "je je pense")
            words = message_lower.split()
            repetitive_hesitations = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])

            # Long pauses indicated by multiple dots or dashes
            pause_hesitations = message_lower.count('...') + message_lower.count('---')

            if direct_hesitations + repetitive_hesitations + pause_hesitations > 0:
                hesitation_count += 1

        return hesitation_count / total_messages

    def _analyze_question_complexity(self, session_data: pd.DataFrame) -> float:
        """Analyze the complexity of questions asked by the user"""
        questions = session_data[session_data['user_input'].str.contains(r'[?]|^(qu|comment|pourquoi|où|quand)', 
                                                                          case=False, na=False)]

        if len(questions) == 0:
            return 0.0

        complexity_scores = []
        for question in questions['user_input']:
            # Length-based complexity
            word_count = len(question.split())
            length_score = min(word_count / 20, 1.0)  # Normalize to 0-1

            # Semantic complexity using embeddings
            embedding = self.embeddings_model.encode([question])
            semantic_complexity = np.linalg.norm(embedding[0])  # Vector magnitude as complexity proxy
            semantic_score = min(semantic_complexity / 10, 1.0)  # Normalize

            # Syntactic complexity (subordinate clauses, conjunctions)
            syntactic_indicators = ['parce que', 'puisque', 'étant donné', 'si', 'quand', 'comme']
            syntactic_score = sum(1 for indicator in syntactic_indicators if indicator in question.lower()) / len(syntactic_indicators)

            complexity_scores.append((length_score + semantic_score + syntactic_score) / 3)

        return np.mean(complexity_scores)

    def _calculate_engagement_score(self, session_data: pd.DataFrame) -> float:
        """Calculate user engagement based on multiple factors"""
        if len(session_data) == 0:
            return 0.0

        # Message frequency
        time_diffs = pd.to_datetime(session_data['timestamp']).diff().dt.total_seconds().fillna(0)
        avg_response_time = np.mean(time_diffs[1:]) if len(time_diffs) > 1 else 0

        # Engagement indicators
        engagement_indicators = [
            'intéressant', 'merci', 'super', 'génial', 'parfait', 
            'j\'aimerais', 'pouvez-vous', 'expliquer', 'détailler'
        ]

        engagement_count = 0
        total_words = 0

        for message in session_data['user_input']:
            message_lower = message.lower()
            total_words += len(message.split())
            engagement_count += sum(1 for indicator in engagement_indicators if indicator in message_lower)

        # Normalize engagement indicators by word count
        engagement_rate = engagement_count / max(total_words, 1)

        # Response time factor (faster responses indicate higher engagement)
        time_factor = max(0, 1 - (avg_response_time / 300))  # 5 minutes as baseline

        # Message length factor (longer messages often indicate engagement)
        avg_message_length = np.mean([len(msg.split()) for msg in session_data['user_input']])
        length_factor = min(avg_message_length / 15, 1.0)  # Normalize to 0-1

        return (engagement_rate + time_factor + length_factor) / 3

    def _analyze_interaction_patterns(self, session_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze patterns in user-bot interactions"""
        patterns = {
            'question_to_statement_ratio': 0.0,
            'followup_question_rate': 0.0,
            'topic_shift_frequency': 0.0,
            'clarification_requests': 0.0
        }

        if len(session_data) == 0:
            return patterns

        # Question to statement ratio
        questions = session_data['user_input'].str.contains(r'[?]', na=False).sum()
        total_messages = len(session_data)
        patterns['question_to_statement_ratio'] = questions / max(total_messages, 1)

        # Follow-up question detection using embeddings
        user_messages = session_data['user_input'].tolist()
        if len(user_messages) > 1:
            embeddings = self.embeddings_model.encode(user_messages)
            similarities = []

            for i in range(1, len(embeddings)):
                similarity = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                similarities.append(similarity)

            # High similarity between consecutive messages indicates follow-up questions
            followup_threshold = 0.7
            followups = sum(1 for sim in similarities if sim > followup_threshold)
            patterns['followup_question_rate'] = followups / max(len(similarities), 1)

            # Topic shifts (low similarity between consecutive messages)
            topic_shifts = sum(1 for sim in similarities if sim < 0.3)
            patterns['topic_shift_frequency'] = topic_shifts / max(len(similarities), 1)

        # Clarification requests
        clarification_phrases = [
            'je ne comprends pas', 'pouvez-vous expliquer', 'que voulez-vous dire',
            'c\'est quoi', 'comment ça', 'répéter', 'préciser'
        ]

        clarification_count = 0
        for message in session_data['user_input']:
            message_lower = message.lower()
            if any(phrase in message_lower for phrase in clarification_phrases):
                clarification_count += 1

        patterns['clarification_requests'] = clarification_count / max(total_messages, 1)

        return patterns

    def _perform_semantic_clustering(self, session_data: pd.DataFrame) -> List[int]:
        """Perform semantic clustering of user messages"""
        if len(session_data) < 2:
            return [0] * len(session_data)

        user_messages = session_data['user_input'].tolist()
        embeddings = self.embeddings_model.encode(user_messages)

        # Determine optimal number of clusters (max 5, min 2)
        n_clusters = min(max(len(user_messages) // 3, 1), 5)

        if n_clusters == 1:
            return [0] * len(user_messages)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        return clusters.tolist()

    def _calculate_learning_pace(self, session_data: pd.DataFrame) -> float:
        """Calculate learning pace based on question evolution and understanding indicators"""
        if len(session_data) < 3:
            return 0.5  # Default moderate pace

        # Analyze question sophistication over time
        user_messages = session_data['user_input'].tolist()
        sophistication_scores = []

        for message in user_messages:
            # Simple sophistication metrics
            word_count = len(message.split())
            unique_words = len(set(message.lower().split()))
            lexical_diversity = unique_words / max(word_count, 1)

            # Technical vocabulary indicators
            technical_terms = ['fonction', 'algorithme', 'variable', 'méthode', 'classe', 'objet']
            technical_score = sum(1 for term in technical_terms if term in message.lower())

            sophistication = (lexical_diversity + technical_score / len(technical_terms)) / 2
            sophistication_scores.append(sophistication)

        # Calculate trend in sophistication (learning pace)
        if len(sophistication_scores) > 1:
            # Simple linear trend
            x = np.arange(len(sophistication_scores))
            coeffs = np.polyfit(x, sophistication_scores, 1)
            pace = max(0, min(1, coeffs[0] + 0.5))  # Normalize to 0-1
        else:
            pace = 0.5

        return pace

    def _detect_confusion_indicators(self, session_data: pd.DataFrame) -> float:
        """Detect indicators of confusion or difficulty"""
        if len(session_data) == 0:
            return 0.0
"""
Simplified behavioral analysis module
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class BehavioralFeatures:
    """Container for behavioral analysis features"""
    session_id: str
    hesitation_rate: float
    question_complexity: float
    engagement_score: float
    interaction_patterns: Dict[str, float]
    semantic_clusters: List[int]
    learning_pace: float
    confusion_indicators: float

class BehavioralAnalyzer:
    """Simplified behavioral analysis without heavy NLP dependencies"""

    def __init__(self):
        self.hesitation_patterns = [
            "euh", "mmh", "hmm", "ben", "alors", "donc",
            "comment dire", "je sais pas", "peut-être", "je pense que",
            "j'hésite", "c'est compliqué", "difficile à dire"
        ]

    def analyze_session_behavior(self, session_data: pd.DataFrame) -> BehavioralFeatures:
        """Comprehensive behavioral analysis for a single session"""
        if session_data.empty:
            return BehavioralFeatures(
                session_id="unknown",
                hesitation_rate=0.0,
                question_complexity=0.0,
                engagement_score=0.0,
                interaction_patterns={},
                semantic_clusters=[],
                learning_pace=0.0,
                confusion_indicators=0.0
            )

        session_id = session_data['session_id'].iloc[0]

        # Calculate various behavioral metrics
        hesitation_rate = self._calculate_hesitation_rate(session_data)
        question_complexity = self._analyze_question_complexity(session_data)
        engagement_score = self._calculate_engagement_score(session_data)
        interaction_patterns = self._analyze_interaction_patterns(session_data)
        semantic_clusters = self._perform_simple_clustering(session_data)
        learning_pace = self._calculate_learning_pace(session_data)
        confusion_indicators = self._detect_confusion_indicators(session_data)

        return BehavioralFeatures(
            session_id=session_id,
            hesitation_rate=hesitation_rate,
            question_complexity=question_complexity,
            engagement_score=engagement_score,
            interaction_patterns=interaction_patterns,
            semantic_clusters=semantic_clusters,
            learning_pace=learning_pace,
            confusion_indicators=confusion_indicators
        )

    def _calculate_hesitation_rate(self, session_data: pd.DataFrame) -> float:
        """Calculate hesitation rate using pattern matching"""
        total_messages = len(session_data)
        if total_messages == 0:
            return 0.0

        hesitation_count = 0
        for _, row in session_data.iterrows():
            message = str(row.get('user_input', '')).lower()

            # Direct pattern matching
            for pattern in self.hesitation_patterns:
                if pattern in message:
                    hesitation_count += 1
                    break

            # Long pauses indicated by multiple dots
            if '...' in message or '---' in message:
                hesitation_count += 1

        return hesitation_count / total_messages

    def _analyze_question_complexity(self, session_data: pd.DataFrame) -> float:
        """Analyze the complexity of questions (simplified)"""
        questions = []
        for _, row in session_data.iterrows():
            message = str(row.get('user_input', ''))
            if '?' in message or any(word in message.lower() for word in ['comment', 'pourquoi', 'qu', 'quel']):
                questions.append(message)

        if not questions:
            return 0.0

        # Simple complexity based on word count and question words
        complexity_scores = []
        for question in questions:
            word_count = len(question.split())
            complexity = min(word_count / 20, 1.0)  # Normalize to 0-1
            complexity_scores.append(complexity)

        return np.mean(complexity_scores)

    def _calculate_engagement_score(self, session_data: pd.DataFrame) -> float:
        """Calculate user engagement based on multiple factors"""
        if session_data.empty:
            return 0.0

        # Message length as proxy for engagement
        message_lengths = []
        for _, row in session_data.iterrows():
            message = str(row.get('user_input', ''))
            message_lengths.append(len(message.split()))

        avg_length = np.mean(message_lengths) if message_lengths else 0
        length_score = min(avg_length / 15, 1.0)  # Normalize

        # Engagement indicators
        engagement_words = ['intéressant', 'merci', 'super', 'génial', 'parfait']
        engagement_count = 0
        total_words = sum(message_lengths)

        for _, row in session_data.iterrows():
            message = str(row.get('user_input', '')).lower()
            engagement_count += sum(1 for word in engagement_words if word in message)

        engagement_rate = engagement_count / max(total_words, 1)

        return (length_score + min(engagement_rate * 10, 1.0)) / 2

    def _analyze_interaction_patterns(self, session_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze patterns in user-bot interactions"""
        if session_data.empty:
            return {}

        total_messages = len(session_data)
        questions = 0
        short_messages = 0
        long_messages = 0

        for _, row in session_data.iterrows():
            message = str(row.get('user_input', ''))
            message_length = len(message)

            if '?' in message:
                questions += 1
            if message_length < 20:
                short_messages += 1
            elif message_length > 100:
                long_messages += 1

        return {
            'question_to_statement_ratio': questions / max(total_messages, 1),
            'short_message_ratio': short_messages / max(total_messages, 1),
            'long_message_ratio': long_messages / max(total_messages, 1)
        }

    def _perform_simple_clustering(self, session_data: pd.DataFrame) -> List[int]:
        """Simple clustering based on message similarity (without embeddings)"""
        if len(session_data) < 2:
            return [0] * len(session_data)

        # Simple clustering based on message length and question content
        clusters = []
        for _, row in session_data.iterrows():
            message = str(row.get('user_input', ''))

            # Simple heuristic clustering
            if '?' in message:
                cluster = 1  # Questions
            elif len(message.split()) > 10:
                cluster = 2  # Long messages
            else:
                cluster = 0  # Short statements

            clusters.append(cluster)

        return clusters

    def _calculate_learning_pace(self, session_data: pd.DataFrame) -> float:
        """Calculate learning pace (simplified)"""
        if len(session_data) < 3:
            return 0.5

        # Simple pace calculation based on message evolution
        message_lengths = []
        for _, row in session_data.iterrows():
            message = str(row.get('user_input', ''))
            message_lengths.append(len(message.split()))

        # Check if messages are getting more sophisticated over time
        first_half = message_lengths[:len(message_lengths)//2]
        second_half = message_lengths[len(message_lengths)//2:]

        if np.mean(second_half) > np.mean(first_half):
            return 0.7  # Increasing complexity
        else:
            return 0.3  # Decreasing or stable complexity

    def _detect_confusion_indicators(self, session_data: pd.DataFrame) -> float:
        """Detect indicators of confusion or difficulty"""
        if session_data.empty:
            return 0.0

        confusion_phrases = [
            'je ne comprends pas', 'c\'est difficile', 'compliqué', 'confus',
            'je n\'y arrive pas', 'ça ne marche pas', 'problème', 'erreur',
            'aide', 'bloqué', 'coincé'
        ]

        confusion_count = 0
        total_messages = len(session_data)

        for _, row in session_data.iterrows():
            message = str(row.get('user_input', '')).lower()
            if any(phrase in message for phrase in confusion_phrases):
                confusion_count += 1

        return min(confusion_count / max(total_messages, 1), 1.0)
        confusion_phrases = [
            'je ne comprends pas', 'c\'est difficile', 'compliqué', 'confus',
            'je n\'y arrive pas', 'ça ne marche pas', 'problème', 'erreur',
            'aide', 'bloqué', 'coincé'
        ]

        confusion_count = 0
        total_messages = len(session_data)

        for message in session_data['user_input']:
            message_lower = message.lower()
            if any(phrase in message_lower for phrase in confusion_phrases):
                confusion_count += 1

        # Also consider repeated questions as confusion indicators
        user_messages = session_data['user_input'].tolist()
        if len(user_messages) > 1:
            embeddings = self.embeddings_model.encode(user_messages)

            repeated_questions = 0
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    if similarity > 0.85:  # High similarity threshold for repetition
                        repeated_questions += 1
                        break

            confusion_count += repeated_questions

        return min(confusion_count / max(total_messages, 1), 1.0)
