from typing import List, Dict
from supabase import Client
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

class OutletMatcher:
    def __init__(self, supabase_client: Client):
        # Initialize spaCy with fallback
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except Exception as e:
            print(f"spaCy not available, using TF-IDF fallback: {str(e)}")
            self.nlp = None
        
        self.supabase = supabase_client
        self._vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=5000)
        
        # Pre-compute outlet embeddings for efficiency
        self._outlet_embeddings = {}
        self._outlet_texts = {}
        self._initialize_outlet_embeddings()

    def _initialize_outlet_embeddings(self):
        """Pre-compute embeddings for all outlets to improve performance."""
        try:
            outlets = self.get_outlets()
            print(f"Initializing embeddings for {len(outlets)} outlets...")
            
            for outlet in outlets:
                outlet_id = outlet.get('id', outlet.get('Outlet Name', ''))
                outlet_text = self._extract_outlet_text(outlet)
                self._outlet_texts[outlet_id] = outlet_text
                
                # Compute embeddings
                if self.nlp:
                    doc = self.nlp(outlet_text)
                    # Convert numpy vector to Python list for JSON serialization
                    vector_list = doc.vector.tolist() if hasattr(doc.vector, 'tolist') else list(doc.vector)
                    self._outlet_embeddings[outlet_id] = {
                        'vector': vector_list,
                        'entities': [ent.text.lower() for ent in doc.ents],
                        'noun_chunks': [chunk.text.lower() for chunk in doc.noun_chunks],
                        'keywords': self._extract_keywords(doc)
                    }
                else:
                    # Fallback to TF-IDF
                    self._outlet_embeddings[outlet_id] = {
                        'text': outlet_text,
                        'keywords': self._extract_fallback_keywords(outlet_text)
                    }
            
            print("Outlet embeddings initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")

    def _extract_outlet_text(self, outlet: Dict) -> str:
        """Extract comprehensive text representation of outlet."""
        fields = [
            outlet.get('Outlet Name', ''),
            outlet.get('Audience', ''),
            outlet.get('Keywords', ''),
            outlet.get('Section Name', ''),
            outlet.get('Pitch Tips', ''),
            outlet.get('Guidelines', '')
        ]
        return ' '.join(filter(None, fields))

    def _extract_keywords(self, doc) -> List[str]:
        """Extract meaningful keywords using spaCy."""
        keywords = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'TECH', 'EDU']:
                keywords.append(ent.text.lower())
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:
                keywords.append(chunk.text.lower())
        
        # Extract important nouns and adjectives
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2 and not token.is_stop:
                keywords.append(token.text.lower())
        
        return list(set(keywords))

    def _extract_fallback_keywords(self, text: str) -> List[str]:
        """Fallback keyword extraction using simple NLP."""
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))

    def find_matches(self, abstract: str, industry: str, limit: int = 20) -> List[Dict]:
        """Find matching outlets using advanced semantic analysis."""
        try:
            outlets = self.get_outlets()
            if not outlets:
                return []
            
            # Create query representation
            query_text = f"{abstract} {industry}"
            
            matches = []
            for outlet in outlets:
                outlet_id = outlet.get('id', outlet.get('Outlet Name', ''))
                
                # Calculate comprehensive similarity score
                score = self._calculate_comprehensive_similarity(
                    query_text, abstract, industry, outlet, outlet_id
                )
                
                # Generate detailed explanation
                explanation = self._generate_match_explanation(
                    query_text, abstract, industry, outlet, outlet_id, score
                )
                
                if score > 0.15:  # Moderate minimum relevance threshold
                    matches.append({
                        "outlet": outlet,
                        "score": self._ensure_json_serializable(round(score, 3)),
                        "match_confidence": f"{round(score * 100)}%",
                        "match_explanation": explanation
                    })
            
            # Sort by score and return top matches
            matches.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"\n📊 MATCHING RESULTS:")
            print(f"   Total outlets processed: {len(outlets)}")
            print(f"   Matches found: {len(matches)}")
            if matches:
                print(f"   Score range: {matches[-1]['score']:.3f} - {matches[0]['score']:.3f}")
            
            return matches[:limit]
            
        except Exception as e:
            print(f"Error in find_matches: {str(e)}")
            return []

    def _calculate_comprehensive_similarity(self, query_text: str, abstract: str, industry: str, outlet: Dict, outlet_id: str) -> float:
        """Calculate comprehensive similarity using multiple advanced techniques."""
        try:
            # 1. Semantic similarity using embeddings
            semantic_score = self._calculate_semantic_similarity_advanced(query_text, outlet_id)
            
            # 2. Industry-specific relevance
            industry_score = self._calculate_industry_relevance_advanced(industry, outlet_id)
            
            # 3. Content relevance
            content_score = self._calculate_content_relevance_advanced(abstract, outlet_id)
            
            # 4. Topic modeling similarity
            topic_score = self._calculate_topic_similarity_advanced(query_text, outlet_id)
            
            # 5. Entity overlap
            entity_score = self._calculate_entity_overlap_advanced(query_text, outlet_id)
            
            # 6. Contextual relevance
            context_score = self._calculate_contextual_relevance_advanced(abstract, industry, outlet_id)
            
            # Weighted combination with dynamic weights based on content type
            weights = self._calculate_dynamic_weights(abstract, industry)
            
            # Calculate base weighted score
            base_weighted_score = (
                semantic_score * weights['semantic'] +
                industry_score * weights['industry'] +
                content_score * weights['content'] +
                topic_score * weights['topic'] +
                entity_score * weights['entity'] +
                context_score * weights['context']
            )
            
            # Apply industry-specific boosting for excellent matches
            industry_boost = self._calculate_industry_boost(industry, outlet_id)
            
            # Apply moderate scaling for better differentiation
            if base_weighted_score > 0.8:
                # Excellent matches get moderate boost
                final_score = base_weighted_score + (base_weighted_score - 0.8) * 1.0 + industry_boost
            elif base_weighted_score > 0.6:
                # Very good matches get slight boost
                final_score = base_weighted_score + (base_weighted_score - 0.6) * 0.5 + industry_boost
            elif base_weighted_score > 0.4:
                # Good matches get minimal boost
                final_score = base_weighted_score + (base_weighted_score - 0.4) * 0.25 + industry_boost
            else:
                # Poor matches get slight boost instead of penalty
                final_score = base_weighted_score * 1.1 + industry_boost
            
            final_score = min(1.0, final_score)
            
            # Apply non-linear transformation for better differentiation
            final_score = self._apply_score_transformation(final_score)
            
            # Ensure we return a Python float
            return float(min(1.0, final_score))
            
        except Exception as e:
            print(f"Error calculating comprehensive similarity: {e}")
            return 0.0

    def _calculate_semantic_similarity_advanced(self, query_text: str, outlet_id: str) -> float:
        """Calculate semantic similarity using advanced NLP techniques."""
        try:
            if self.nlp and outlet_id in self._outlet_embeddings:
                # Use spaCy vectors for semantic similarity
                query_doc = self.nlp(query_text.lower())
                outlet_vector = self._outlet_embeddings[outlet_id]['vector']
                
                # Convert outlet_vector back to numpy array if it's a list
                if isinstance(outlet_vector, list):
                    outlet_vector = np.array(outlet_vector)
                
                # Calculate cosine similarity between vectors
                similarity = np.dot(query_doc.vector, outlet_vector) / (
                    np.linalg.norm(query_doc.vector) * np.linalg.norm(outlet_vector)
                )
                
                # Convert numpy float to Python float
                return float(max(0.0, similarity))
            else:
                # Fallback to TF-IDF
                return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))
                
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0

    def _calculate_industry_relevance_advanced(self, industry: str, outlet_id: str) -> float:
        """Calculate industry relevance using advanced techniques."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id].lower()
            industry_lower = industry.lower()
            
            # 1. Direct keyword matching (higher weight for exact matches)
            keyword_score = self._calculate_keyword_overlap(industry_lower, outlet_text)
            
            # 2. Enhanced semantic similarity
            semantic_score = self._calculate_tfidf_similarity(industry_lower, outlet_text)
            
            # 3. Entity matching
            entity_score = self._calculate_entity_matching(industry_lower, outlet_id)
            
            # 4. Contextual relevance
            context_score = self._calculate_contextual_industry_relevance(industry_lower, outlet_text)
            
            # 5. Industry-specific keyword matching
            industry_keyword_score = self._calculate_industry_specific_keywords(industry_lower, outlet_text)
            
            # Weighted combination with higher emphasis on keyword matching
            base_score = (
                keyword_score * 0.35 + 
                semantic_score * 0.25 + 
                entity_score * 0.15 + 
                context_score * 0.15 + 
                industry_keyword_score * 0.10
            )
            
            # Apply moderate non-linear scaling for better differentiation
            if base_score > 0.8:
                return 0.85 + (base_score - 0.8) * 0.75  # 85-100% for excellent industry matches
            elif base_score > 0.6:
                return 0.70 + (base_score - 0.6) * 0.75  # 70-85% for very good matches
            elif base_score > 0.4:
                return 0.55 + (base_score - 0.4) * 0.75  # 55-70% for good matches
            elif base_score > 0.2:
                return 0.40 + (base_score - 0.2) * 0.75  # 40-55% for moderate matches
            elif base_score > 0.1:
                return 0.25 + (base_score - 0.1) * 1.5   # 25-40% for weak matches
            else:
                return base_score * 2.5  # 0-25% for poor matches
            
        except Exception as e:
            print(f"Error in industry relevance: {e}")
            return 0.0

    def _calculate_content_relevance_advanced(self, abstract: str, outlet_id: str) -> float:
        """Calculate content relevance using advanced NLP."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id]
            
            # 1. Topic similarity
            topic_score = self._calculate_topic_similarity_advanced(abstract, outlet_id)
            
            # 2. Keyword overlap
            keyword_score = self._calculate_keyword_overlap(abstract.lower(), outlet_text.lower())
            
            # 3. Semantic similarity
            semantic_score = self._calculate_tfidf_similarity(abstract, outlet_text)
            
            # 4. Technical term matching
            tech_score = self._calculate_technical_term_matching(abstract, outlet_text)
            
            # Weighted combination
            base_score = (topic_score * 0.3 + keyword_score * 0.25 + semantic_score * 0.25 + tech_score * 0.2)
            
            # Apply non-linear scaling for better differentiation
            if base_score > 0.8:
                return 0.85 + (base_score - 0.8) * 0.75  # 85-92% for excellent content matches
            elif base_score > 0.6:
                return 0.65 + (base_score - 0.6) * 1.0   # 65-85% for very good matches
            elif base_score > 0.4:
                return 0.45 + (base_score - 0.4) * 1.0   # 45-65% for good matches
            elif base_score > 0.2:
                return 0.25 + (base_score - 0.2) * 1.0   # 25-45% for moderate matches
            else:
                return base_score * 1.25  # 0-25% for weak matches
            
        except Exception as e:
            print(f"Error in content relevance: {e}")
            return 0.0

    def _calculate_topic_similarity_advanced(self, query_text: str, outlet_id: str) -> float:
        """Calculate topic similarity using advanced topic modeling."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            # Extract topics from query and outlet
            query_topics = self._extract_topics_advanced(query_text)
            outlet_topics = self._outlet_embeddings[outlet_id].get('keywords', [])
            
            if not query_topics or not outlet_topics:
                return 0.0
            
            # Calculate Jaccard similarity
            query_set = set(query_topics)
            outlet_set = set(outlet_topics)
            
            intersection = query_set.intersection(outlet_set)
            union = query_set.union(outlet_set)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            print(f"Error in topic similarity: {e}")
            return 0.0

    def _calculate_entity_overlap_advanced(self, query_text: str, outlet_id: str) -> float:
        """Calculate entity overlap using advanced entity recognition."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            # Extract entities from query
            if self.nlp:
                query_doc = self.nlp(query_text.lower())
                query_entities = set([ent.text.lower() for ent in query_doc.ents])
            else:
                query_entities = set(self._extract_fallback_keywords(query_text))
            
            # Get outlet entities
            outlet_entities = set(self._outlet_embeddings[outlet_id].get('entities', []))
            
            if not query_entities or not outlet_entities:
                return 0.0
            
            # Calculate overlap
            intersection = query_entities.intersection(outlet_entities)
            union = query_entities.union(outlet_entities)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception as e:
            print(f"Error in entity overlap: {e}")
            return 0.0

    def _calculate_contextual_relevance_advanced(self, abstract: str, industry: str, outlet_id: str) -> float:
        """Calculate contextual relevance using advanced context analysis."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id]
            
            # 1. Industry context matching
            industry_context = self._extract_industry_context(industry)
            industry_context_score = self._calculate_context_matching(industry_context, outlet_text)
            
            # 2. Abstract context matching
            abstract_context = self._extract_abstract_context(abstract)
            abstract_context_score = self._calculate_context_matching(abstract_context, outlet_text)
            
            # 3. Combined context analysis
            combined_context = f"{abstract} {industry}"
            combined_context_score = self._calculate_context_matching(combined_context, outlet_text)
            
            # Weighted combination
            return (industry_context_score * 0.4 + abstract_context_score * 0.3 + combined_context_score * 0.3)
            
        except Exception as e:
            print(f"Error in contextual relevance: {e}")
            return 0.0

    def _calculate_dynamic_weights(self, abstract: str, industry: str) -> Dict[str, float]:
        """Calculate dynamic weights based on content characteristics."""
        # Analyze content type and adjust weights accordingly
        abstract_lower = abstract.lower()
        industry_lower = industry.lower()
        
        # Default weights with higher emphasis on industry relevance
        weights = {
            'semantic': 0.20,
            'industry': 0.35,  # Higher weight for industry relevance
            'content': 0.20,
            'topic': 0.15,
            'entity': 0.05,
            'context': 0.05
        }
        
        # Adjust based on content characteristics
        if any(tech_term in abstract_lower for tech_term in ['ai', 'artificial intelligence', 'machine learning', 'technology']):
            weights['content'] += 0.05
            weights['topic'] += 0.05
            weights['semantic'] -= 0.05
            weights['industry'] -= 0.05
        
        # Special handling for education industry
        if any(edu_term in industry_lower for edu_term in ['education', 'learning', 'academic', 'teaching', 'school', 'university']):
            weights['industry'] += 0.10  # Much higher weight for education
            weights['context'] += 0.05
            weights['content'] -= 0.05
            weights['topic'] -= 0.05
            weights['semantic'] -= 0.05
        
        # Special handling for healthcare industry
        if any(health_term in industry_lower for health_term in ['healthcare', 'health', 'medical', 'patient']):
            weights['industry'] += 0.10
            weights['context'] += 0.05
            weights['content'] -= 0.05
            weights['topic'] -= 0.05
            weights['semantic'] -= 0.05
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights

    def _apply_score_transformation(self, score: float) -> float:
        """Apply non-linear transformation for better score differentiation."""
        if score > 0.8:
            result = 0.85 + (score - 0.8) * 0.75  # 85-100% for excellent matches
        elif score > 0.6:
            result = 0.70 + (score - 0.6) * 0.75  # 70-85% for very good matches
        elif score > 0.4:
            result = 0.50 + (score - 0.4) * 1.0   # 50-70% for good matches
        elif score > 0.2:
            result = 0.30 + (score - 0.2) * 1.0   # 30-50% for moderate matches
        elif score > 0.1:
            result = 0.15 + (score - 0.1) * 1.5   # 15-30% for weak matches
        else:
            result = score * 1.5  # 0-15% for poor matches
        
        # Ensure we return a Python float
        return float(result)

    def _extract_topics_advanced(self, text: str) -> List[str]:
        """Extract topics using advanced NLP techniques."""
        try:
            if self.nlp:
                doc = self.nlp(text.lower())
                topics = []
                
                # Extract noun phrases
                for chunk in doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:
                        topics.append(chunk.text.lower())
                
                # Extract named entities
                for ent in doc.ents:
                    topics.append(ent.text.lower())
                
                # Extract important words
                for token in doc:
                    if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 2 and not token.is_stop:
                        topics.append(token.text.lower())
                
                return list(set(topics))
            else:
                return self._extract_fallback_keywords(text)
                
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []

    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            words1 = words1 - stop_words
            words2 = words2 - stop_words
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            base_score = len(intersection) / len(union)
            
            # Boost score for important industry terms
            important_terms = {
                'education', 'educational', 'learning', 'teaching', 'academic', 'school',
                'university', 'college', 'student', 'teacher', 'professor', 'curriculum',
                'healthcare', 'health', 'medical', 'patient', 'clinical', 'hospital',
                'technology', 'tech', 'software', 'ai', 'artificial intelligence'
            }
            
            important_matches = intersection.intersection(important_terms)
            if important_matches:
                # Boost score for important term matches
                boost = len(important_matches) * 0.1
                base_score = min(1.0, base_score + boost)
            
            return base_score
            
        except Exception as e:
            print(f"Error in keyword overlap: {e}")
            return 0.0

    def _calculate_entity_matching(self, industry: str, outlet_id: str) -> float:
        """Calculate entity matching between industry and outlet."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_entities = set(self._outlet_embeddings[outlet_id].get('entities', []))
            industry_words = set(industry.lower().split())
            
            if not outlet_entities or not industry_words:
                return 0.0
            
            # Find matching entities
            matches = 0
            for word in industry_words:
                if len(word) > 2 and any(word in entity for entity in outlet_entities):
                    matches += 1
            
            return matches / len(industry_words) if industry_words else 0.0
            
        except Exception as e:
            print(f"Error in entity matching: {e}")
            return 0.0

    def _calculate_contextual_industry_relevance(self, industry: str, outlet_text: str) -> float:
        """Calculate contextual industry relevance."""
        try:
            # Extract industry-related context
            industry_context = self._extract_industry_context(industry)
            return self._calculate_context_matching(industry_context, outlet_text)
            
        except Exception as e:
            print(f"Error in contextual industry relevance: {e}")
            return 0.0

    def _extract_industry_context(self, industry: str) -> str:
        """Extract industry context and related terms."""
        # This could be expanded with industry-specific knowledge bases
        industry_lower = industry.lower()
        
        # Add related terms based on industry
        context_terms = []
        if 'education' in industry_lower:
            context_terms.extend(['learning', 'teaching', 'academic', 'student', 'curriculum'])
        elif 'technology' in industry_lower:
            context_terms.extend(['innovation', 'digital', 'software', 'platform'])
        elif 'healthcare' in industry_lower:
            context_terms.extend(['medical', 'health', 'patient', 'clinical'])
        
        return f"{industry} {' '.join(context_terms)}"

    def _extract_abstract_context(self, abstract: str) -> str:
        """Extract context from abstract."""
        # Extract key concepts and themes
        if self.nlp:
            doc = self.nlp(abstract.lower())
            context_terms = []
            
            for token in doc:
                if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 3 and not token.is_stop:
                    context_terms.append(token.text)
            
            return f"{abstract} {' '.join(context_terms[:10])}"
        else:
            return abstract

    def _calculate_context_matching(self, context: str, outlet_text: str) -> float:
        """Calculate context matching between context and outlet text."""
        try:
            return self._calculate_tfidf_similarity(context, outlet_text)
        except Exception as e:
            print(f"Error in context matching: {e}")
            return 0.0

    def _calculate_technical_term_matching(self, abstract: str, outlet_text: str) -> float:
        """Calculate technical term matching."""
        try:
            tech_terms = [
                'ai', 'artificial intelligence', 'machine learning', 'platform', 'software',
                'technology', 'digital', 'innovation', 'solution', 'system', 'data',
                'cloud', 'cybersecurity', 'blockchain', 'api', 'algorithm'
            ]
            
            abstract_lower = abstract.lower()
            outlet_lower = outlet_text.lower()
            
            matching_terms = [term for term in tech_terms if term in abstract_lower and term in outlet_lower]
            
            return len(matching_terms) / len(tech_terms) if tech_terms else 0.0
            
        except Exception as e:
            print(f"Error in technical term matching: {e}")
            return 0.0

    def _calculate_industry_specific_keywords(self, industry: str, outlet_text: str) -> float:
        """Calculate industry-specific keyword matching."""
        try:
            industry_lower = industry.lower()
            
            # Define industry-specific keyword mappings
            industry_keywords = {
                'education': [
                    'education', 'educational', 'learning', 'teaching', 'academic', 'school',
                    'university', 'college', 'student', 'teacher', 'professor', 'curriculum',
                    'pedagogy', 'instruction', 'tutoring', 'assessment', 'policy', 'leaders',
                    'edtech', 'edutech', 'k12', 'k-12', 'higher education', 'primary education',
                    'secondary education', 'vocational', 'training', 'skills', 'knowledge',
                    'classroom', 'campus', 'faculty', 'administration', 'e-learning', 'online learning',
                    'distance learning', 'blended learning', 'adaptive learning', 'personalized learning',
                    'educational technology', 'learning management system', 'lms', 'mooc', 'course',
                    'lesson', 'syllabus', 'degree', 'diploma', 'certificate', 'accreditation'
                ],
                'healthcare': [
                    'healthcare', 'health', 'medical', 'patient', 'clinical', 'hospital',
                    'doctor', 'physician', 'nurse', 'treatment', 'diagnosis', 'therapy',
                    'medicine', 'pharmaceutical', 'biotech', 'telemedicine', 'digital health',
                    'health it', 'ehr', 'electronic health record', 'patient care', 'wellness'
                ],
                'technology': [
                    'technology', 'tech', 'software', 'hardware', 'digital', 'innovation',
                    'ai', 'artificial intelligence', 'machine learning', 'data science',
                    'cybersecurity', 'cloud computing', 'blockchain', 'iot', 'internet of things',
                    'automation', 'robotics', 'virtual reality', 'augmented reality', 'vr', 'ar'
                ],
                'finance': [
                    'finance', 'financial', 'banking', 'investment', 'fintech', 'payments',
                    'cryptocurrency', 'blockchain', 'trading', 'wealth management', 'insurance',
                    'lending', 'credit', 'debit', 'mobile payments', 'digital banking'
                ],
                'marketing': [
                    'marketing', 'advertising', 'brand', 'campaign', 'digital marketing',
                    'social media', 'content marketing', 'seo', 'sem', 'ppc', 'email marketing',
                    'influencer', 'analytics', 'conversion', 'lead generation', 'customer acquisition'
                ]
            }
            
            # Find matching industry
            matching_industry = None
            for ind, keywords in industry_keywords.items():
                if ind in industry_lower or any(keyword in industry_lower for keyword in keywords):
                    matching_industry = ind
                    break
            
            if not matching_industry:
                # Default to general matching
                return self._calculate_keyword_overlap(industry_lower, outlet_text)
            
            # Calculate keyword matching for the specific industry
            keywords = industry_keywords[matching_industry]
            matching_keywords = [keyword for keyword in keywords if keyword in outlet_text]
            
            # Calculate score based on keyword density
            if matching_keywords:
                # Count occurrences of matching keywords
                total_matches = sum(outlet_text.count(keyword) for keyword in matching_keywords)
                # Normalize by text length and number of keywords
                score = min(1.0, total_matches / (len(outlet_text.split()) * 0.1))
                return score
            
            return 0.0
            
        except Exception as e:
            print(f"Error in industry-specific keyword matching: {e}")
            return 0.0

    def _calculate_industry_boost(self, industry: str, outlet_id: str) -> float:
        """Calculate industry-specific boost for excellent matches."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id].lower()
            industry_lower = industry.lower()
            
            # Define high-value industry keywords that deserve extra boost
            high_value_keywords = {
                'education': [
                    'education', 'educational', 'learning', 'teaching', 'academic', 'school',
                    'university', 'college', 'student', 'teacher', 'professor', 'curriculum',
                    'pedagogy', 'instruction', 'tutoring', 'assessment', 'policy', 'leaders',
                    'edtech', 'edutech', 'k12', 'k-12', 'higher education', 'primary education',
                    'secondary education', 'vocational', 'training', 'skills', 'knowledge',
                    'classroom', 'campus', 'faculty', 'administration', 'e-learning', 'online learning',
                    'distance learning', 'blended learning', 'adaptive learning', 'personalized learning',
                    'educational technology', 'learning management system', 'lms', 'mooc', 'course',
                    'lesson', 'syllabus', 'degree', 'diploma', 'certificate', 'accreditation'
                ],
                'healthcare': [
                    'healthcare', 'health', 'medical', 'patient', 'clinical', 'hospital',
                    'doctor', 'physician', 'nurse', 'treatment', 'diagnosis', 'therapy',
                    'medicine', 'pharmaceutical', 'biotech', 'telemedicine', 'digital health',
                    'health it', 'ehr', 'electronic health record', 'patient care', 'wellness'
                ],
                'technology': [
                    'technology', 'tech', 'software', 'hardware', 'digital', 'innovation',
                    'ai', 'artificial intelligence', 'machine learning', 'data science',
                    'cybersecurity', 'cloud computing', 'blockchain', 'iot', 'internet of things',
                    'automation', 'robotics', 'virtual reality', 'augmented reality', 'vr', 'ar'
                ],
                'finance': [
                    'finance', 'financial', 'banking', 'investment', 'fintech', 'payments',
                    'cryptocurrency', 'blockchain', 'trading', 'wealth management', 'insurance',
                    'lending', 'credit', 'debit', 'mobile payments', 'digital banking'
                ],
                'marketing': [
                    'marketing', 'advertising', 'brand', 'campaign', 'digital marketing',
                    'social media', 'content marketing', 'seo', 'sem', 'ppc', 'email marketing',
                    'influencer', 'analytics', 'conversion', 'lead generation', 'customer acquisition'
                ]
            }
            
            # Find matching industry
            matching_industry = None
            for ind, keywords in high_value_keywords.items():
                if ind in industry_lower or any(keyword in industry_lower for keyword in keywords):
                    matching_industry = ind
                    break
            
            if not matching_industry:
                return 0.0
            
            # Calculate boost based on high-value keyword matches
            keywords = high_value_keywords[matching_industry]
            matching_keywords = [keyword for keyword in keywords if keyword in outlet_text]
            
            if matching_keywords:
                # Count high-value keyword occurrences
                total_matches = sum(outlet_text.count(keyword) for keyword in matching_keywords)
                # Calculate boost (0.10 to 0.25 for excellent matches)
                boost = min(0.25, total_matches * 0.03)
                return boost
            
            return 0.0
            
        except Exception as e:
            print(f"Error in industry boost calculation: {e}")
            return 0.0

    def _generate_match_explanation(self, query_text: str, abstract: str, industry: str, outlet: Dict, outlet_id: str, score: float) -> List[str]:
        """Generate detailed match explanation using advanced analysis."""
        try:
            explanation = []
            
            # Add score-based explanation
            if score > 0.9:
                explanation.append("Perfect match with comprehensive industry and content alignment")
            elif score > 0.8:
                explanation.append("Excellent match with high relevance across all dimensions")
            elif score > 0.6:
                explanation.append("Strong match with good alignment to content and industry")
            elif score > 0.4:
                explanation.append("Good match with moderate relevance indicators")
            elif score > 0.2:
                explanation.append("Moderate match with some relevant connections")
            else:
                explanation.append("Limited match with minimal relevance detected")
            
            # Add specific matching details
            if outlet_id in self._outlet_embeddings:
                outlet_text = self._outlet_texts[outlet_id]
                
                # Industry matching details
                industry_score = self._calculate_industry_relevance_advanced(industry, outlet_id)
                if industry_score > 0.3:
                    explanation.append(f"Industry alignment: {round(industry_score * 100)}%")
                
                # Content matching details
                content_score = self._calculate_content_relevance_advanced(abstract, outlet_id)
                if content_score > 0.3:
                    explanation.append(f"Content relevance: {round(content_score * 100)}%")
                
                # Topic matching details
                topic_score = self._calculate_topic_similarity_advanced(query_text, outlet_id)
                if topic_score > 0.2:
                    explanation.append(f"Topic similarity: {round(topic_score * 100)}%")
                
                # Entity matching details
                entity_score = self._calculate_entity_overlap_advanced(query_text, outlet_id)
                if entity_score > 0.2:
                    explanation.append(f"Entity overlap: {round(entity_score * 100)}%")
            
            # Add outlet characteristics
            if outlet.get('AI Partnered', ''):
                explanation.append("AI Partnered outlet")
            
            if outlet.get('Prestige', ''):
                explanation.append(f"Prestige level: {outlet.get('Prestige', '')}")
            
            # Add audience information
            if outlet.get('Audience', ''):
                explanation.append(f"Target audience: {outlet.get('Audience', '')}")
            
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return ["Match analysis completed"]

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using TF-IDF vectors."""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            texts = [self._clean_text(text1), self._clean_text(text2)]
            tfidf_matrix = self._vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity = similarity_matrix[0][0]
            # Ensure we return a Python float, not numpy float
            return float(similarity)
        except Exception as e:
            print(f"Error in TF-IDF similarity: {e}")
            return 0.0

    def _clean_text(self, text: str) -> str:
        """Clean text for better matching."""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable by converting numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        else:
            return obj

    def get_outlets(self) -> List[Dict]:
        """Fetch outlets from database."""
        try:
            response = self.supabase.table("outlets").select("*").execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error fetching outlets: {str(e)}")
            return []

   