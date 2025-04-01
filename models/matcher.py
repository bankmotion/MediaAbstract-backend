import os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet
import nltk
from typing import List, Dict, Tuple
from supabase import Client

class OutletMatcher:
    def __init__(self, supabase_client: Client):
        # Use more lenient settings for TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.95
        )
        self.supabase = supabase_client
        self.threshold = 0.01  # Lower threshold for matching

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        if not text:
            return ""
        
        # Convert to lowercase and tokenize
        words = nltk.word_tokenize(text.lower())
        expanded_words = set()
        
        # Add original words
        for word in words:
            expanded_words.add(word)
            
            # Add synonyms
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded_words.add(lemma.name().lower())
                    
                # Add hypernyms (more general terms)
                for hypernym in syn.hypernyms():
                    for lemma in hypernym.lemmas():
                        expanded_words.add(lemma.name().lower())
        
        return ' '.join(expanded_words)

    def calculate_similarity_score(self, outlet: Dict, query: str) -> float:
        """Calculate similarity score between outlet and query."""
        try:
            # Combine outlet fields with weights
            outlet_parts = []
            
            # Keywords get highest weight (x3)
            if outlet.get('Keywords'):
                keywords = outlet.get('Keywords', '').lower()
                outlet_parts.extend([keywords] * 3)
            
            # Audience gets second highest weight (x2)    
            if outlet.get('Audience'):
                audience = outlet.get('Audience', '').lower()
                outlet_parts.extend([audience] * 2)
            
            # Other fields get normal weight
            if outlet.get('Section Name'):
                section = outlet.get('Section Name', '').lower()
                outlet_parts.append(section)
            
            if outlet.get('Outlet Name'):
                name = outlet.get('Outlet Name', '').lower()
                outlet_parts.append(name)
            
            outlet_text = ' '.join(outlet_parts)
            query = query.lower()
            
            # Print raw text for debugging
            print(f"\nRaw outlet text: {outlet_text}")
            print(f"Query text: {query}")
            
            # Create word sets
            outlet_words = set(outlet_text.split())
            query_words = set(query.split())
            
            # Calculate Jaccard similarity
            intersection = len(outlet_words.intersection(query_words))
            union = len(outlet_words.union(query_words))
            
            if union == 0:
                return 0.0
            
            similarity = intersection / union
            
            # Add bonus for exact matches
            for word in query_words:
                if word in outlet_text:
                    similarity += 0.1
            
            print(f"Matching words: {outlet_words.intersection(query_words)}")
            print(f"Similarity score: {similarity}")
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            print(f"Outlet text: {outlet_text}")
            print(f"Query text: {query}")
            return 0.0

    def find_matches(self, query: str) -> List[Dict]:
        """Find matching outlets with debug information."""
        try:
            outlets = self.get_outlets()
            if not outlets:
                print("No outlets found in database")
                return []
                
            matches = []
            print(f"Processing query: {query}")
            print(f"Found {len(outlets)} outlets to match against")
            
            for outlet in outlets:
                score = self.calculate_similarity_score(outlet, query)
                print(f"Outlet: {outlet.get('Outlet Name')} - Score: {score}")
                
                if score >= self.threshold:
                    matches.append({
                        "outlet": outlet,
                        "score": score,
                        "match_confidence": round(score * 100, 2),
                        "match_explanation": self._generate_match_explanation(outlet, score, query)
                    })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            # Return all matches, not just top 10
            print(f"Found {len(matches)} matches above threshold {self.threshold}")
            return matches
            
        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            return []

    def _generate_match_explanation(self, outlet: Dict, score: float, query: str) -> str:
        """Generate more detailed match explanations."""
        reasons = []
        
        # Score-based explanation
        if score >= 0.5:
            reasons.append("Strong overall match")
        elif score >= 0.3:
            reasons.append("Good match")
        elif score >= 0.1:
            reasons.append("Partial match")
        else:
            reasons.append("Weak match but potentially relevant")
        
        # Content-based explanations
        if outlet.get('Keywords'):
            common_terms = set(query.lower().split()) & set(outlet['Keywords'].lower().split())
            if common_terms:
                reasons.append(f"Matching keywords: {', '.join(common_terms)}")
        
        if outlet.get('Audience'):
            if outlet['Audience'].lower() in query.lower():
                reasons.append(f"Matches target audience: {outlet['Audience']}")
        
        return "; ".join(reasons)

    def get_outlets(self) -> List[Dict]:
        """Fetch outlets with error handling."""
        try:
            response = self.supabase.table("outlets").select("*").execute()
            if not response.data:
                print("No outlets found in database")
                return []
            
            # Print sample of outlets data
            print("\nSample outlet data:")
            for outlet in response.data[:2]:  # Print first 2 outlets
                print(f"Outlet: {outlet}")
                
            return response.data
        except Exception as e:
            print(f"Error fetching outlets: {str(e)}")
            return []