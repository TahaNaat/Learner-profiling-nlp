import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging

# Optional imports with fallbacks
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Some features may be limited.")

logger = logging.getLogger(__name__)

@dataclass
class LearningStyle:
    """VARK learning style representation"""
    visual: float = 0.25
    auditory: float = 0.25
    reading: float = 0.25
    kinesthetic: float = 0.25

@dataclass
class PersonalityProfile:
    """MBTI-inspired personality dimensions"""
    extraversion: float = 0.5  # vs introversion
    intuition: float = 0.5     # vs sensing
    thinking: float = 0.5      # vs feeling
    judging: float = 0.5       # vs perceiving

@dataclass
class CognitiveProfile:
    """Cognitive abilities and preferences"""
    processing_speed: float = 0.5
    working_memory: float = 0.5
    analytical_thinking: float = 0.5
    creative_thinking: float = 0.5
    attention_span: float = 0.5

@dataclass
class BehavioralProfile:
    """Behavioral patterns in learning"""
    engagement_level: float = 0.5
    persistence: float = 0.5
    help_seeking: float = 0.5
    collaboration_preference: float = 0.5
    self_regulation: float = 0.5

@dataclass
class LearnerProfile:
    """Comprehensive learner profile"""
    learner_id: str
    learning_style: LearningStyle
    personality: PersonalityProfile
    cognitive: CognitiveProfile
    behavioral: BehavioralProfile
    difficulty_areas: List[str]
    strengths: List[str]
    recommended_strategies: List[str]
    confidence_level: float = 0.5
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class ProfileGenerator:
    """Generate and update learner profiles dynamically"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.profile_cache: Dict[str, LearnerProfile] = {}

    def generate_profile(self, 
                        session_features: pd.DataFrame,
                        questionnaire_data: Optional[pd.DataFrame] = None,
                        behavioral_features: Optional[List[Any]] = None) -> LearnerProfile:
        """Generate comprehensive learner profile from conversation data only"""

        learner_id = session_features['session_id'].iloc[0] if not session_features.empty else "unknown"

        # Extract learning style from conversation patterns ONLY
        learning_style = self._infer_learning_style_from_conversations(session_features)

        # Infer personality traits from interaction patterns
        personality = self._infer_personality_traits(session_features)

        # Assess cognitive abilities
        cognitive = self._assess_cognitive_profile(session_features)

        # Analyze behavioral patterns
        behavioral = self._analyze_behavioral_profile(session_features, behavioral_features)

        # Identify strengths and difficulties
        strengths, difficulties = self._identify_strengths_difficulties(session_features)

        # Generate personalized recommendations
        recommendations = self._generate_recommendations(learning_style, personality, cognitive, behavioral)

        # Calculate overall confidence level
        confidence = self._calculate_confidence_level(session_features)

        profile = LearnerProfile(
            learner_id=learner_id,
            learning_style=learning_style,
            personality=personality,
            cognitive=cognitive,
            behavioral=behavioral,
            difficulty_areas=difficulties,
            strengths=strengths,
            recommended_strategies=recommendations,
            confidence_level=confidence
        )

        self.profile_cache[learner_id] = profile
        return profile

    def _infer_learning_style_from_conversations(self, session_features: pd.DataFrame) -> LearningStyle:
        """Infer VARK learning style ONLY from conversation patterns"""

        # Initialize counters
        visual_indicators = 0
        auditory_indicators = 0
        reading_indicators = 0
        kinesthetic_indicators = 0

        total_words = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            words = user_input.split()
            total_words += len(words)

            # Visual learning indicators (plus complets)
            visual_words = [
                'voir', 'regarder', 'image', 'schéma', 'graphique', 'couleur', 'visuel',
                'diagramme', 'carte', 'tableau', 'figure', 'illustration', 'montrer',
                'afficher', 'visualiser', 'observer'
            ]
            visual_indicators += sum(1 for word in visual_words if word in user_input)

            # Auditory learning indicators
            auditory_words = [
                'entendre', 'écouter', 'son', 'audio', 'expliquer', 'dire', 'parler',
                'voix', 'prononcer', 'répéter', 'discuter', 'conversation', 'oral'
            ]
            auditory_indicators += sum(1 for word in auditory_words if word in user_input)

            # Reading/Writing indicators
            reading_words = [
                'lire', 'écrire', 'texte', 'livre', 'documentation', 'article',
                'notes', 'résumé', 'rédiger', 'script', 'manuel', 'guide'
            ]
            reading_indicators += sum(1 for word in reading_words if word in user_input)

            # Kinesthetic indicators
            kinesthetic_words = [
                'faire', 'pratiquer', 'essayer', 'manipuler', 'exercice', 'action',
                'bouger', 'toucher', 'construire', 'créer', 'expérimenter', 'tester'
            ]
            kinesthetic_indicators += sum(1 for word in kinesthetic_words if word in user_input)

        # Normalize by total words to get proportional scores
        if total_words > 0:
            visual_score = visual_indicators / total_words
            auditory_score = auditory_indicators / total_words
            reading_score = reading_indicators / total_words
            kinesthetic_score = kinesthetic_indicators / total_words
        else:
            # Default balanced profile if no indicators found
            visual_score = auditory_score = reading_score = kinesthetic_score = 0.25

        # Ensure scores sum to 1.0 and have minimum baseline
        total_score = visual_score + auditory_score + reading_score + kinesthetic_score

        if total_score > 0:
            # Normalize to sum to 1.0
            visual_score /= total_score
            auditory_score /= total_score
            reading_score /= total_score
            kinesthetic_score /= total_score
        else:
            # If no specific indicators, assume balanced learning style
            visual_score = auditory_score = reading_score = kinesthetic_score = 0.25

        return LearningStyle(
            visual=visual_score,
            auditory=auditory_score,
            reading=reading_score,
            kinesthetic=kinesthetic_score
        )


    def _infer_personality_traits(self, session_features: pd.DataFrame) -> PersonalityProfile:
        """Infer personality traits from interaction patterns"""

        if session_features.empty:
            return PersonalityProfile()

        # Extraversion indicators
        social_words = ['partager', 'équipe', 'groupe', 'collaboration', 'discussion']
        extraversion_score = 0

        # Intuition indicators
        abstract_words = ['concept', 'théorie', 'idée', 'innovation', 'créatif', 'possibilité']
        intuition_score = 0

        # Thinking indicators
        logical_words = ['logique', 'analyse', 'raisonnement', 'objectif', 'rationnel']
        thinking_score = 0

        # Judging indicators
        structure_words = ['planifier', 'organiser', 'structure', 'méthode', 'étapes']
        judging_score = 0

        total_messages = len(session_features)

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()

            extraversion_score += sum(1 for word in social_words if word in user_input)
            intuition_score += sum(1 for word in abstract_words if word in user_input)
            thinking_score += sum(1 for word in logical_words if word in user_input)
            judging_score += sum(1 for word in structure_words if word in user_input)

        # Normalize scores
        extraversion = min(extraversion_score / max(total_messages, 1) + 0.5, 1.0)
        intuition = min(intuition_score / max(total_messages, 1) + 0.5, 1.0)
        thinking = min(thinking_score / max(total_messages, 1) + 0.5, 1.0)
        judging = min(judging_score / max(total_messages, 1) + 0.5, 1.0)

        return PersonalityProfile(
            extraversion=extraversion,
            intuition=intuition,
            thinking=thinking,
            judging=judging
        )

    def _assess_cognitive_profile(self, session_features: pd.DataFrame) -> CognitiveProfile:
        """Assess cognitive abilities from interaction patterns"""

        if session_features.empty:
            return CognitiveProfile()

        # Processing speed (based on response times)
        if 'time_diff_sec' in session_features.columns:
            avg_response_time = session_features['time_diff_sec'].mean()
            processing_speed = max(0, 1 - (avg_response_time / 120))  # 2 minutes baseline
        else:
            processing_speed = 0.5

        # Working memory (based on question complexity and follow-ups)
        complex_questions = 0
        total_questions = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', ''))
            if '?' in user_input:
                total_questions += 1
                if len(user_input.split()) > 10:  # Complex question threshold
                    complex_questions += 1

        working_memory = complex_questions / max(total_questions, 1)

        # Analytical thinking (based on analytical language)
        analytical_words = ['analyse', 'compare', 'évalue', 'critique', 'examine']
        analytical_count = 0
        total_words = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            words = user_input.split()
            total_words += len(words)
            analytical_count += sum(1 for word in analytical_words if word in words)

        analytical_thinking = analytical_count / max(total_words, 1) * 100  # Scale up

        # Creative thinking (based on creative language)
        creative_words = ['créatif', 'innovation', 'original', 'nouveau', 'différent']
        creative_count = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            creative_count += sum(1 for word in creative_words if word in user_input)

        creative_thinking = creative_count / max(len(session_features), 1)

        # Attention span (based on session length and engagement)
        attention_span = min(len(session_features) / 20, 1.0)  # Normalize by expected session length

        return CognitiveProfile(
            processing_speed=min(processing_speed, 1.0),
            working_memory=min(working_memory, 1.0),
            analytical_thinking=min(analytical_thinking, 1.0),
            creative_thinking=min(creative_thinking, 1.0),
            attention_span=attention_span
        )

    def _analyze_behavioral_profile(self, 
                                  session_features: pd.DataFrame,
                                  behavioral_features: Optional[List[Any]] = None) -> BehavioralProfile:
        """Analyze behavioral patterns from session data"""

        if session_features.empty:
            return BehavioralProfile()

        # Engagement level (based on message length and frequency)
        avg_message_length = session_features['user_input'].str.len().mean()
        engagement_level = min(avg_message_length / 100, 1.0)  # Normalize

        # Persistence (based on follow-up questions and continued interaction)
        persistence = min(len(session_features) / 15, 1.0)  # More messages = more persistence

        # Help-seeking behavior
        help_words = ['aide', 'aidez-moi', 'pouvez-vous', 'expliquer', 'comment']
        help_seeking_count = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            if any(word in user_input for word in help_words):
                help_seeking_count += 1

        help_seeking = help_seeking_count / max(len(session_features), 1)

        # Collaboration preference (inferred from language)
        collab_words = ['ensemble', 'équipe', 'partager', 'collaboration']
        collab_count = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            collab_count += sum(1 for word in collab_words if word in user_input)

        collaboration_preference = collab_count / max(len(session_features), 1)

        # Self-regulation (based on metacognitive language)
        meta_words = ['je pense', 'je crois', 'je comprends', 'je réfléchis']
        meta_count = 0

        for _, row in session_features.iterrows():
            user_input = str(row.get('user_input', '')).lower()
            meta_count += sum(1 for phrase in meta_words if phrase in user_input)

        self_regulation = meta_count / max(len(session_features), 1)

        return BehavioralProfile(
            engagement_level=min(engagement_level, 1.0),
            persistence=min(persistence, 1.0),
            help_seeking=min(help_seeking, 1.0),
            collaboration_preference=min(collaboration_preference, 1.0),
            self_regulation=min(self_regulation, 1.0)
        )

    def _identify_strengths_difficulties(self, session_features: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify learner strengths and difficulty areas"""

        strengths = []
        difficulties = []

        if session_features.empty:
            return strengths, difficulties

        # Analyze sentiment patterns
        if 'sentiment_score' in session_features.columns:
            avg_sentiment = session_features['sentiment_score'].mean()
            if avg_sentiment > 3.5:
                strengths.append("Attitude positive")
            elif avg_sentiment < 2.5:
                difficulties.append("Motivation ou moral bas")

        # Analyze comprehension patterns
        if 'comprehension_score' in session_features.columns:
            avg_comprehension = session_features['comprehension_score'].mean()
            if avg_comprehension > 0.7:
                strengths.append("Bonne compréhension")
            elif avg_comprehension < 0.3:
                difficulties.append("Difficultés de compréhension")

        # Analyze question patterns
        question_count = session_features['user_input'].str.contains(r'\?').sum()
        if question_count > len(session_features) * 0.6:
            strengths.append("Curiosité et engagement actif")
        elif question_count < len(session_features) * 0.2:
            difficulties.append("Faible engagement interrogatif")

        # Analyze hesitation patterns
        if 'has_hesitation' in session_features.columns:
            hesitation_rate = session_features['has_hesitation'].mean()
            if hesitation_rate < 0.2:
                strengths.append("Communication claire et assurée")
            elif hesitation_rate > 0.5:
                difficulties.append("Hésitations fréquentes")

        return strengths, difficulties

    def _generate_recommendations(self, 
                                learning_style: LearningStyle,
                                personality: PersonalityProfile,
                                cognitive: CognitiveProfile,
                                behavioral: BehavioralProfile) -> List[str]:
        """Generate personalized learning recommendations"""

        recommendations = []

        # Learning style recommendations
        if learning_style.visual > 0.4:
            recommendations.append("Utiliser des diagrammes, graphiques et supports visuels")
        if learning_style.auditory > 0.4:
            recommendations.append("Privilégier les explications orales et discussions")
        if learning_style.reading > 0.4:
            recommendations.append("Fournir des ressources textuelles et documentation écrite")
        if learning_style.kinesthetic > 0.4:
            recommendations.append("Proposer des exercices pratiques et manipulations")

        # Personality-based recommendations
        if personality.extraversion > 0.6:
            recommendations.append("Encourager le travail de groupe et discussions")
        else:
            recommendations.append("Respecter le besoin de temps de réflexion individuelle")

        if personality.intuition > 0.6:
            recommendations.append("Présenter les concepts théoriques et connexions globales")
        else:
            recommendations.append("Fournir des exemples concrets et étapes détaillées")

        # Cognitive recommendations
        if cognitive.processing_speed < 0.4:
            recommendations.append("Laisser plus de temps pour l'assimilation")
        if cognitive.working_memory < 0.4:
            recommendations.append("Décomposer les informations en petites unités")

        # Behavioral recommendations
        if behavioral.help_seeking < 0.3:
            recommendations.append("Encourager à poser des questions et demander de l'aide")
        if behavioral.persistence < 0.4:
            recommendations.append("Proposer des objectifs courts et réalisables")

        return recommendations

    def _calculate_confidence_level(self, session_features: pd.DataFrame) -> float:
        """Calculate overall confidence level in the profile"""

        if session_features.empty:
            return 0.3  # Low confidence for empty data

        # Factors contributing to confidence
        data_completeness = min(len(session_features) / 10, 1.0)  # More data = higher confidence

        # Consistency in patterns (less variance = higher confidence)
        if 'sentiment_score' in session_features.columns:
            sentiment_variance = session_features['sentiment_score'].var()
            sentiment_consistency = max(0, 1 - (sentiment_variance / 2))
        else:
            sentiment_consistency = 0.5

        # Time span of data collection
        if 'timestamp' in session_features.columns:
            time_span = (session_features['timestamp'].max() - session_features['timestamp'].min()).total_seconds()
            time_factor = min(time_span / (7 * 24 * 3600), 1.0)  # Week as optimal span
        else:
            time_factor = 0.5

        confidence = (data_completeness + sentiment_consistency + time_factor) / 3
        return min(max(confidence, 0.1), 0.95)  # Keep between 10% and 95%

    def update_profile(self, learner_id: str, new_session_data: pd.DataFrame) -> LearnerProfile:
        """Update existing learner profile with new session data"""

        if learner_id in self.profile_cache:
            existing_profile = self.profile_cache[learner_id]

            # Generate new profile features
            new_profile = self.generate_profile(new_session_data)

            # Weighted average with existing profile (70% existing, 30% new)
            updated_profile = self._merge_profiles(existing_profile, new_profile, weight=0.3)
            updated_profile.last_updated = datetime.now()

            self.profile_cache[learner_id] = updated_profile
            return updated_profile
        else:
            return self.generate_profile(new_session_data)

    def _merge_profiles(self, existing: LearnerProfile, new: LearnerProfile, weight: float = 0.3) -> LearnerProfile:
        """Merge existing and new profiles with weighted average"""

        # Merge learning styles
        merged_learning_style = LearningStyle(
            visual=existing.learning_style.visual * (1-weight) + new.learning_style.visual * weight,
            auditory=existing.learning_style.auditory * (1-weight) + new.learning_style.auditory * weight,
            reading=existing.learning_style.reading * (1-weight) + new.learning_style.reading * weight,
            kinesthetic=existing.learning_style.kinesthetic * (1-weight) + new.learning_style.kinesthetic * weight
        )

        # Merge personality traits
        merged_personality = PersonalityProfile(
            extraversion=existing.personality.extraversion * (1-weight) + new.personality.extraversion * weight,
            intuition=existing.personality.intuition * (1-weight) + new.personality.intuition * weight,
            thinking=existing.personality.thinking * (1-weight) + new.personality.thinking * weight,
            judging=existing.personality.judging * (1-weight) + new.personality.judging * weight
        )

        # Merge cognitive profile
        merged_cognitive = CognitiveProfile(
            processing_speed=existing.cognitive.processing_speed * (1-weight) + new.cognitive.processing_speed * weight,
            working_memory=existing.cognitive.working_memory * (1-weight) + new.cognitive.working_memory * weight,
            analytical_thinking=existing.cognitive.analytical_thinking * (1-weight) + new.cognitive.analytical_thinking * weight,
            creative_thinking=existing.cognitive.creative_thinking * (1-weight) + new.cognitive.creative_thinking * weight,
            attention_span=existing.cognitive.attention_span * (1-weight) + new.cognitive.attention_span * weight
        )

        # Merge behavioral profile
        merged_behavioral = BehavioralProfile(
            engagement_level=existing.behavioral.engagement_level * (1-weight) + new.behavioral.engagement_level * weight,
            persistence=existing.behavioral.persistence * (1-weight) + new.behavioral.persistence * weight,
            help_seeking=existing.behavioral.help_seeking * (1-weight) + new.behavioral.help_seeking * weight,
            collaboration_preference=existing.behavioral.collaboration_preference * (1-weight) + new.behavioral.collaboration_preference * weight,
            self_regulation=existing.behavioral.self_regulation * (1-weight) + new.behavioral.self_regulation * weight
        )

        # Combine lists (strengths, difficulties, recommendations)
        merged_strengths = list(set(existing.strengths + new.strengths))
        merged_difficulties = list(set(existing.difficulty_areas + new.difficulty_areas))
        merged_recommendations = list(set(existing.recommended_strategies + new.recommended_strategies))

        # Update confidence
        merged_confidence = existing.confidence_level * (1-weight) + new.confidence_level * weight

        return LearnerProfile(
            learner_id=existing.learner_id,
            learning_style=merged_learning_style,
            personality=merged_personality,
            cognitive=merged_cognitive,
            behavioral=merged_behavioral,
            difficulty_areas=merged_difficulties,
            strengths=merged_strengths,
            recommended_strategies=merged_recommendations,
            confidence_level=merged_confidence
        )

    def export_profile(self, learner_id: str, format: str = 'json') -> str:
        """Export learner profile in specified format"""

        if learner_id not in self.profile_cache:
            return ""

        profile = self.profile_cache[learner_id]

        if format.lower() == 'json':
            profile_dict = asdict(profile)
            profile_dict['last_updated'] = profile.last_updated.isoformat()
            return json.dumps(profile_dict, indent=2, ensure_ascii=False)

        return ""

    def get_similar_learners(self, learner_id: str, n_similar: int = 5) -> List[str]:
        """Find similar learners based on profile characteristics"""

        if learner_id not in self.profile_cache or len(self.profile_cache) < 2:
            return []

        target_profile = self.profile_cache[learner_id]
        similarities = []

        for other_id, other_profile in self.profile_cache.items():
            if other_id == learner_id:
                continue

            # Calculate similarity based on multiple dimensions
            similarity = self._calculate_profile_similarity(target_profile, other_profile)
            similarities.append((other_id, similarity))

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [learner_id for learner_id, _ in similarities[:n_similar]]

    def _calculate_profile_similarity(self, profile1: LearnerProfile, profile2: LearnerProfile) -> float:
        """Calculate similarity between two learner profiles"""

        # Learning style similarity
        ls1 = [profile1.learning_style.visual, profile1.learning_style.auditory, 
               profile1.learning_style.reading, profile1.learning_style.kinesthetic]
        ls2 = [profile2.learning_style.visual, profile2.learning_style.auditory,
               profile2.learning_style.reading, profile2.learning_style.kinesthetic]

        ls_similarity = 1 - np.linalg.norm(np.array(ls1) - np.array(ls2)) / 2

        # Personality similarity
        p1 = [profile1.personality.extraversion, profile1.personality.intuition,
              profile1.personality.thinking, profile1.personality.judging]
        p2 = [profile2.personality.extraversion, profile2.personality.intuition,
              profile2.personality.thinking, profile2.personality.judging]

        p_similarity = 1 - np.linalg.norm(np.array(p1) - np.array(p2)) / 2

        # Cognitive similarity
        c1 = [profile1.cognitive.processing_speed, profile1.cognitive.working_memory,
              profile1.cognitive.analytical_thinking, profile1.cognitive.creative_thinking,
              profile1.cognitive.attention_span]
        c2 = [profile2.cognitive.processing_speed, profile2.cognitive.working_memory,
              profile2.cognitive.analytical_thinking, profile2.cognitive.creative_thinking,
              profile2.cognitive.attention_span]

        c_similarity = 1 - np.linalg.norm(np.array(c1) - np.array(c2)) / np.sqrt(5)

        # Behavioral similarity
        b1 = [profile1.behavioral.engagement_level, profile1.behavioral.persistence,
              profile1.behavioral.help_seeking, profile1.behavioral.collaboration_preference,
              profile1.behavioral.self_regulation]
        b2 = [profile2.behavioral.engagement_level, profile2.behavioral.persistence,
              profile2.behavioral.help_seeking, profile2.behavioral.collaboration_preference,
              profile2.behavioral.self_regulation]

        b_similarity = 1 - np.linalg.norm(np.array(b1) - np.array(b2)) / np.sqrt(5)

        # Weighted average
        overall_similarity = (ls_similarity * 0.3 + p_similarity * 0.25 + 
                            c_similarity * 0.25 + b_similarity * 0.2)

        return max(0, min(1, overall_similarity))
