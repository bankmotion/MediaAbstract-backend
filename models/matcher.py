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
        """Calculate similarity score using spaCy's semantic similarity."""
        try:
            # Process query and outlet text with spaCy
            query_doc = self.nlp(query.lower())
            
            # Combine outlet fields with weights
            outlet_parts = []
            weighted_scores = []
            
            # Process each field with its weight
            for field, weight in self.field_weights.items():
                if outlet.get(field):
                    field_text = outlet.get(field, '').lower()
                    field_doc = self.nlp(field_text)
                    
                    # Calculate semantic similarity
                    similarity = query_doc.similarity(field_doc)
                    weighted_scores.append(similarity * weight)
                    
                    # Add to outlet parts for Jaccard similarity
                    outlet_parts.extend([field_text] * int(weight))
            
            # Calculate weighted average of semantic similarities
            if weighted_scores:
                semantic_score = sum(weighted_scores) / sum(self.field_weights.values())
            else:
                semantic_score = 0.0
            
            # Calculate Jaccard similarity as a fallback
            outlet_text = self.preprocess_text(' '.join(outlet_parts))
            query_text = self.preprocess_text(query)
            
            outlet_words = set(outlet_text.split())
            query_words = set(query_text.split())
            
            intersection = len(outlet_words.intersection(query_words))
            union = len(outlet_words.union(query_words))
            
            jaccard_score = intersection / union if union > 0 else 0.0
            
            # Combine scores (70% semantic, 30% Jaccard)
            final_score = (0.7 * semantic_score) + (0.3 * jaccard_score)
            
            # Add bonus for exact matches
            for word in query_words:
                if word in outlet_text:
                    final_score += 0.1
            
            return min(1.0, final_score)  # Cap at 1.0
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def find_matches(self, query: str, limit: int = 20) -> List[Dict]:
        """Find matching outlets with improved scoring and limit."""
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
                        "match_confidence": round(score * 100, 2),
                        "match_explanation": self._generate_match_explanation(outlet, score, query)
                    })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            # Return top matches
            print(f"Found {len(matches)} matches above threshold {self.threshold}")
            return matches[:limit]
            
        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            return []

    def _generate_match_explanation(self, outlet: Dict, score: float, query: str) -> str:
        """Generate detailed match explanations with semantic matching details."""
        reasons = []
        
        # Score-based explanation
        if score >= 0.7:
            reasons.append("Excellent semantic match")
        elif score >= 0.5:
            reasons.append("Strong semantic match")
        elif score >= 0.3:
            reasons.append("Good semantic match")
        else:
            reasons.append("Relevant match based on keywords")
        
        # Process query and outlet text with spaCy
        query_doc = self.nlp(query.lower())
        
        # Analyze each field
        for field, weight in self.field_weights.items():
            if outlet.get(field):
                field_text = outlet.get(field, '').lower()
                field_doc = self.nlp(field_text)
                
                # Find matching entities
                query_entities = {ent.text.lower() for ent in query_doc.ents}
                field_entities = {ent.text.lower() for ent in field_doc.ents}
                matching_entities = query_entities & field_entities
                
                if matching_entities:
                    reasons.append(f"Matching {field} entities: {', '.join(matching_entities)}")
                
                # Find similar terms using spaCy
                similar_terms = []
                for query_token in query_doc:
                    if not query_token.is_stop and not query_token.is_punct:
                        for field_token in field_doc:
                            if not field_token.is_stop and not field_token.is_punct:
                                if query_token.similarity(field_token) > 0.7:
                                    similar_terms.append(f"{query_token.text} ≈ {field_token.text}")
                
                if similar_terms:
                    reasons.append(f"Similar terms in {field}: {', '.join(similar_terms[:3])}")
        
        # Add community notes if available
        if self.feedback_data:
            outlet_feedback = [f for f in self.feedback_data if f.get('outlet_id') == outlet.get('id')]
            if outlet_feedback:
                success_rate = sum(1 for f in outlet_feedback if f.get('success')) / len(outlet_feedback)
                reasons.append(f"Community success rate: {success_rate:.0%}")
                
                # Add recent notes
                recent_notes = [f.get('notes') for f in outlet_feedback if f.get('notes')][-2:]
                if recent_notes:
                    reasons.append(f"Recent community notes: {'; '.join(recent_notes)}")
        
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