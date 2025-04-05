import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import nltk
from typing import List, Dict, Tuple, Optional
from supabase import Client
import spacy
import numpy as np
from datetime import datetime

class OutletMatcher:
    def __init__(self, supabase_client: Client):
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_md")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        self.supabase = supabase_client
        self.threshold = 0.3  # Increased threshold as requested
        
        # Default field weights
        self.field_weights = {
            'Keywords': 3.0,
            'Audience': 2.0,
            'Section Name': 1.0,
            'Outlet Name': 1.0
        }
        
        # Load user feedback data
        self._load_feedback_data()

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with NER."""
        if not text:
            return ""
        
        # Process with spaCy
        doc = self.nlp(text.lower())
        
        # Extract named entities and important terms
        entities = [ent.text for ent in doc.ents]
        important_terms = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        # Combine with original text
        processed_text = ' '.join([text.lower()] + entities + important_terms)
        
        # Add WordNet synonyms
        words = nltk.word_tokenize(processed_text)
        expanded_words = set(words)
        
        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_words.add(lemma.name().lower())
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        expanded_words.add(lemma.name().lower())
        
        return ' '.join(expanded_words)

    def calculate_similarity_score(self, outlet: Dict, query: str) -> float:
        """Calculate similarity score with stricter semantic filtering, Jaccard fallback, audience penalty, and exact match clamping."""
        try:
            query_doc = self.nlp(query.lower())
            query_text = self.preprocess_text(query)
            query_words = set(query_text.split())
            
            outlet_parts = []
            weighted_scores = []

            for field, weight in self.field_weights.items():
                field_text = outlet.get(field, '').lower()
                if field_text:
                    field_doc = self.nlp(field_text)
                    similarity = query_doc.similarity(field_doc)
                    
                    # Only include similarity if it's reasonably strong
                    if similarity > 0.6:
                        weighted_scores.append(min(similarity, 0.95) * weight)
                        outlet_parts.extend([field_text] * int(weight))

            # Weighted semantic score
            semantic_score = (
                sum(weighted_scores) / sum(self.field_weights.values())
                if weighted_scores else 0.0
            )

            # Jaccard similarity
            outlet_text = self.preprocess_text(' '.join(outlet_parts))
            outlet_words = set(outlet_text.split())
            jaccard_score = (
                len(query_words & outlet_words) / len(query_words | outlet_words)
                if query_words | outlet_words else 0.0
            )

            # Final score: 70% semantic, 30% Jaccard
            final_score = (0.7 * semantic_score) + (0.3 * jaccard_score)

            # Cap exact match bonus
            exact_match_bonus = min(0.1, sum(0.01 for word in query_words if word in outlet_text))
            final_score += exact_match_bonus

            # Penalize if Audience mismatch
            query_audience = "tech" if "tech" in query.lower() else None
            outlet_audience = outlet.get('Audience', '').lower()
            if query_audience and query_audience not in outlet_audience:
                final_score = max(0.0, final_score - 0.3)

            return min(1.0, final_score)

        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_matches(self, query: str, limit: int = 20) -> List[Dict]:
        """Find and normalize matches, and provide clean match explanations."""
        try:
            outlets = self.get_outlets()
            if not outlets:
                print("No outlets found in database")
                return []

            matches = []
            for outlet in outlets:
                score = self.calculate_similarity_score(outlet, query)
                if score >= self.threshold:
                    matches.append({
                        "outlet": outlet,
                        "score": score,
                        "match_confidence": 0.0,  # To be filled after normalization
                        "match_explanation": self._generate_match_explanation(outlet, score, query)
                    })

            # Normalize scores
            if matches:
                max_score = max(m['score'] for m in matches)
                for m in matches:
                    norm_score = m['score'] / max_score if max_score > 0 else 0
                    m['score'] = round(norm_score, 4)
                    m['match_confidence'] = round(norm_score * 100, 2)

            matches.sort(key=lambda x: x["score"], reverse=True)
            print(f"Found {len(matches)} matches above threshold {self.threshold}")
            return matches[:limit]

        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            return []

    def _generate_match_explanation(self, outlet: Dict, score: float, query: str) -> str:
        """Generate structured, short explanations for matches."""
        reasons = []
        query_doc = self.nlp(query.lower())

        # General match quality
        if score >= 0.8:
            reasons.append("Excellent match")
        elif score >= 0.6:
            reasons.append("Strong match")
        elif score >= 0.4:
            reasons.append("Relevant match")
        else:
            reasons.append("Keyword-based match")

        # Entity and token similarity
        for field in self.field_weights:
            field_text = outlet.get(field, '').lower()
            if field_text:
                field_doc = self.nlp(field_text)

                # Named entity match
                query_entities = {ent.text.lower() for ent in query_doc.ents}
                field_entities = {ent.text.lower() for ent in field_doc.ents}
                matches = query_entities & field_entities
                if matches:
                    reasons.append(f"{field}: entity match ({', '.join(matches)})")

                # Token similarity
                similar_terms = []
                for qt in query_doc:
                    if not qt.is_stop and not qt.is_punct:
                        for ft in field_doc:
                            if not ft.is_stop and not ft.is_punct:
                                if qt.similarity(ft) > 0.7:
                                    similar_terms.append(f"{qt.text} ≈ {ft.text}")
                if similar_terms:
                    reasons.append(f"{field}: similar terms ({', '.join(similar_terms[:3])})")

        # Audience logic
        query_audience = "tech" if "tech" in query.lower() else None
        outlet_audience = outlet.get("Audience", "").lower()
        if query_audience:
            if query_audience in outlet_audience:
                reasons.append("Audience match: Tech")
            else:
                reasons.append("Audience mismatch: not Tech")

        # Community data
        if self.feedback_data:
            fb = [f for f in self.feedback_data if f.get('outlet_id') == outlet.get('id')]
            if fb:
                success_rate = sum(1 for f in fb if f.get('success')) / len(fb)
                reasons.append(f"Community success rate: {round(success_rate * 100)}%")
                notes = [f['notes'] for f in fb if f.get('notes')]
                if notes:
                    reasons.append(f"Notes: {'; '.join(notes[-2:])}")

        return "; ".join(reasons)

    def get_outlets(self) -> List[Dict]:
        """Fetch outlets with error handling."""
        try:
            response = self.supabase.table("outlets").select("*").execute()
            if not response.data:
                print("No outlets found in database")
                return []
                
            return response.data
        except Exception as e:
            print(f"Error fetching outlets: {str(e)}")
            return []

    def _load_feedback_data(self):
        """Load user feedback data from Supabase."""
        try:
            response = self.supabase.table("outlet_feedback").select("*").execute()
            self.feedback_data = response.data if response.data else []
            
            # Update weights based on feedback
            self._update_weights_from_feedback()
        except Exception as e:
            print(f"Error loading feedback data: {str(e)}")
            self.feedback_data = []

    def _update_weights_from_feedback(self):
        """Update field weights based on user feedback."""
        if not self.feedback_data:
            return
            
        # Calculate success rates for each field
        field_success = {field: 0 for field in self.field_weights}
        field_total = {field: 0 for field in self.field_weights}
        
        for feedback in self.feedback_data:
            if feedback.get('success'):
                for field in self.field_weights:
                    if feedback.get(field):
                        field_success[field] += 1
                        field_total[field] += 1
            else:
                for field in self.field_weights:
                    if feedback.get(field):
                        field_total[field] += 1
        
        # Update weights based on success rates
        for field in self.field_weights:
            if field_total[field] > 0:
                success_rate = field_success[field] / field_total[field]
                # Adjust weight based on success rate (1.0 to 3.0 range)
                self.field_weights[field] = 1.0 + (success_rate * 2.0)

    def update_field_weights(self, new_weights: Dict[str, float]):
        """Allow users to manually adjust field weights."""
        for field, weight in new_weights.items():
            if field in self.field_weights:
                self.field_weights[field] = max(0.1, min(5.0, weight))  # Clamp between 0.1 and 5.0

    def add_feedback(self, outlet_id: str, success: bool, notes: Optional[str] = None):
        """Add new feedback to the system."""
        try:
            feedback = {
                'outlet_id': outlet_id,
                'success': success,
                'notes': notes,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.table("outlet_feedback").insert(feedback).execute()
            self._load_feedback_data()  # Reload feedback data
        except Exception as e:
            print(f"Error adding feedback: {str(e)}")