from typing import List, Dict, Set, Tuple
from supabase import Client
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import math

class OutletMatcher:
    """Advanced outlet matching system with optimized scoring and filtering."""
    
    # Configuration constants
    TOPIC_OVERLAP_THRESHOLD = 0.60  # Reduced from 0.70 per client feedback
    NICHE_KEYWORD_WEIGHT = 0.50     # Reduced from 0.70 for more conservative scoring
    AUDIENCE_WEIGHT = 4.0           # Reduced from 6.0 for more conservative scoring
    MIN_SCORE_THRESHOLD = 0.15      # Minimum relevance threshold
    
    # Industry-specific exclusions (outlets that shouldn't appear for certain industries)
    INDUSTRY_EXCLUSIONS = {
        'cybersecurity': {
            'Modern Healthcare', 'Payments Dive', 'MedCity News', 'Factor This!',
            'GreenBiz', 'Construction Dive', 'Search Engine Land', 'The Decoder'
        },
        'education': {
            'Modern Healthcare', 'Payments Dive', 'MedCity News', 'Factor This!',
            'GreenBiz', 'Construction Dive', 'BleepingComputer', 'The Hacker News',
            'Security Boulevard', 'Dark Reading', 'TechTalks'
        },
        'healthcare': {
            'BleepingComputer', 'The Hacker News', 'Security Boulevard', 'Dark Reading',
            'TechTalks', 'Search Engine Land', 'The Decoder'
        }
    }
    
    # Industry keyword mappings for enhanced matching
    INDUSTRY_KEYWORDS = {
        'cybersecurity': {
            'cybersecurity', 'security', 'cyber', 'hacking', 'malware', 'ransomware',
            'breach', 'vulnerability', 'threat', 'attack', 'firewall', 'encryption',
            'compliance', 'gdpr', 'sox', 'pci', 'zero-day', 'phishing', 'social engineering',
            'penetration testing', 'incident response', 'siem', 'edr', 'xdr', 'mdr',
            'identity management', 'access control', 'authentication', 'authorization'
        },
        'education': {
            'education', 'educational', 'learning', 'teaching', 'academic', 'school',
            'university', 'college', 'student', 'teacher', 'professor', 'curriculum',
            'pedagogy', 'instruction', 'tutoring', 'assessment', 'policy', 'leaders',
            'edtech', 'edutech', 'k12', 'k-12', 'higher education', 'primary education',
            'secondary education', 'vocational', 'training', 'skills', 'knowledge',
            'classroom', 'campus', 'faculty', 'administration', 'e-learning', 'online learning',
            'distance learning', 'blended learning', 'adaptive learning', 'personalized learning',
            'educational technology', 'learning management system', 'lms', 'mooc', 'course',
            'lesson', 'syllabus', 'degree', 'diploma', 'certificate', 'accreditation'
        },
        'healthcare': {
            'healthcare', 'health', 'medical', 'patient', 'clinical', 'hospital',
            'doctor', 'physician', 'nurse', 'treatment', 'diagnosis', 'therapy',
            'medicine', 'pharmaceutical', 'biotech', 'telemedicine', 'digital health',
            'health it', 'ehr', 'electronic health record', 'patient care', 'wellness'
        }
    }

    def __init__(self, supabase_client: Client):
        """Initialize the outlet matcher with optimized configuration."""
        self.supabase = supabase_client
        self._vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 3), 
            max_features=5000
        )
        
        # Initialize NLP with fallback
        self.nlp = self._initialize_nlp()
        
        # Pre-computed data for performance
        self._outlet_embeddings = {}
        self._outlet_texts = {}
        self._outlet_keywords = {}
        self._outlet_audiences = {}
        
        # Initialize outlet data
        self._initialize_outlet_data()

    def _initialize_nlp(self):
        """Initialize NLP with graceful fallback."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
            return nlp
        except Exception as e:
            print(f"⚠️ spaCy not available, using TF-IDF fallback: {str(e)}")
            return None

    def _initialize_outlet_data(self):
        """Pre-compute outlet data for optimal performance."""
        try:
            outlets = self.get_outlets()
            print(f"🔄 Initializing data for {len(outlets)} outlets...")
            
            valid_outlets = 0
            skipped_outlets = 0
            
            for outlet in outlets:
                outlet_id = outlet.get('id', outlet.get('Outlet Name', ''))
                
                # Skip outlets with no valid ID
                if not outlet_id:
                    skipped_outlets += 1
                    continue
                
                # Parse semicolon-separated fields
                self._outlet_keywords[outlet_id] = self._parse_semicolon_field(
                    outlet.get('Keywords', '')
                )
                self._outlet_audiences[outlet_id] = self._parse_semicolon_field(
                    outlet.get('Audience', '')
                )
                
                # Extract comprehensive text representation
                outlet_text = self._extract_outlet_text(outlet)
                
                # Skip outlets with no meaningful text
                if not outlet_text.strip():
                    skipped_outlets += 1
                    continue
                
                self._outlet_texts[outlet_id] = outlet_text
                
                # Compute embeddings
                self._outlet_embeddings[outlet_id] = self._compute_outlet_embedding(
                    outlet_text, outlet_id
                )
                
                valid_outlets += 1
            
            print(f"✅ Outlet data initialized successfully")
            print(f"   Valid outlets: {valid_outlets}")
            print(f"   Skipped outlets: {skipped_outlets}")
            
        except Exception as e:
            print(f"❌ Error initializing outlet data: {e}")

    def _parse_semicolon_field(self, field_value: str) -> List[str]:
        """Parse semicolon-separated field values into clean arrays."""
        if not field_value:
            return []
        
        # Split by semicolon and clean each value
        values = [value.strip().lower() for value in field_value.split(';')]
        return [value for value in values if value]

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

    def _compute_outlet_embedding(self, outlet_text: str, outlet_id: str) -> Dict:
        """Compute outlet embedding using available NLP."""
        try:
            if self.nlp and outlet_text.strip():
                doc = self.nlp(outlet_text)
                # Ensure vector is not empty
                if hasattr(doc.vector, 'size') and doc.vector.size > 0:
                    vector = doc.vector.tolist() if hasattr(doc.vector, 'tolist') else list(doc.vector)
                else:
                    # Fallback to TF-IDF if vector is empty
                    vector = None
                
                return {
                    'vector': vector,
                    'entities': [ent.text.lower() for ent in doc.ents],
                    'noun_chunks': [chunk.text.lower() for chunk in doc.noun_chunks],
                    'keywords': self._extract_keywords(doc)
                }
            else:
                return {
                    'vector': None,
                    'text': outlet_text,
                    'keywords': self._extract_fallback_keywords(outlet_text)
                }
        except Exception as e:
            print(f"⚠️ Error computing embedding for outlet {outlet_id}: {e}")
            return {
                'vector': None,
                'text': outlet_text,
                'keywords': self._extract_fallback_keywords(outlet_text)
            }

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

    def _compute_match_components(self, query_text: str, abstract: str, industry: str, outlet_id: str) -> Dict[str, float]:
        """Compute all component scores used for matching without any final transformation.
        Returns a dict with semantic, industry, content, topic, niche, audience, boost, weighted, and base.
        """
        # Core components
        semantic_score = self._calculate_semantic_similarity(query_text, outlet_id)
        industry_score = self._calculate_industry_relevance(industry, outlet_id)
        content_score = self._calculate_content_relevance(abstract, outlet_id)
        topic_score = self._calculate_topic_similarity(query_text, outlet_id)
        niche_score = self._calculate_niche_keyword_matching(industry, outlet_id)
        audience_score = self._calculate_audience_relevance(industry, outlet_id)

        # Dynamic weights based on content
        weights = self._calculate_optimized_weights(abstract, industry)

        weighted = (
            semantic_score * weights['semantic'] +
            industry_score * weights['industry'] +
            content_score * weights['content'] +
            topic_score * weights['topic'] +
            niche_score * self.NICHE_KEYWORD_WEIGHT +
            audience_score * self.AUDIENCE_WEIGHT
        )

        boost = self._calculate_industry_boost(industry, outlet_id)
        base = max(0.0, weighted + boost)  # raw, untransformed

        return {
            'semantic': float(semantic_score),
            'industry': float(industry_score),
            'content': float(content_score),
            'topic': float(topic_score),
            'niche': float(niche_score),
            'audience': float(audience_score),
            'boost': float(boost),
            'weighted': float(weighted),
            'base': float(base)
        }

    def _apply_distribution_curve(self, normalized_value: float) -> float:
        """Map a [0,1] normalized score into a calibrated display range with smooth spread.
        Uses a slight power curve to emphasize separation at the top without bunching to 100%.
        Output range: ~0.32 - 0.84 by default.
        """
        x = max(0.0, min(1.0, normalized_value))
        # Power curve for smoother top-end separation
        curved = x ** 0.92
        # Map into display band
        return 0.32 + 0.52 * curved

    def find_matches(self, abstract: str, industry: str, limit: int = 20) -> List[Dict]:
        """Find matching outlets using optimized semantic analysis with batch normalization.
        Ensures scores are well distributed and deterministically sorted.
        """
        try:
            outlets = self.get_outlets()
            if not outlets:
                return []

            # Determine industry category for exclusions
            industry_category = self._categorize_industry(industry)
            excluded_outlets = self.INDUSTRY_EXCLUSIONS.get(industry_category, set())

            # Create query representation
            query_text = f"{abstract} {industry}"

            # First pass: compute raw component scores for all valid outlets
            scored_rows: List[Dict] = []
            for outlet in outlets:
                outlet_name = outlet.get('Outlet Name', '')
                if outlet_name in excluded_outlets:
                    continue

                outlet_id = outlet.get('id', outlet_name)
                if not outlet_id or outlet_id not in self._outlet_texts:
                    continue

                components = self._compute_match_components(query_text, abstract, industry, outlet_id)
                # Guard against NaNs/invalid values
                base = components['base'] if np.isfinite(components['base']) else 0.0

                scored_rows.append({
                    'outlet': outlet,
                    'outlet_id': outlet_id,
                    'name': outlet_name,
                    'components': components,
                    'base': base
                })

            if not scored_rows:
                return []

            # Batch normalization to spread scores evenly
            bases = [row['base'] for row in scored_rows]
            min_base = min(bases)
            max_base = max(bases)
            spread = max_base - min_base

            # If spread is tiny, use composite ranking to create separation
            if spread < 1e-6:
                # Composite rank emphasizing niche and industry, then content/topic
                def composite_key(row):
                    c = row['components']
                    return (
                        c['niche'] * 0.5 + c['industry'] * 0.3 + c['content'] * 0.15 + c['topic'] * 0.05
                    )
                scored_rows.sort(key=composite_key, reverse=True)
                total = len(scored_rows)
                for idx, row in enumerate(scored_rows):
                    rank = idx / max(1, total - 1)
                    normalized = 1.0 - rank  # top gets 1.0
                    adjusted = self._apply_distribution_curve(normalized)
                    # Deterministic tie-breaker using niche/audience micro-adjustment
                    micro = (row['components']['niche'] * 0.5 + row['components']['audience'] * 0.5) * 0.01
                    row['final_score'] = float(min(0.85, adjusted + micro))
            else:
                # Standard normalization path
                for row in scored_rows:
                    normalized = (row['base'] - min_base) / (spread + 1e-9)
                    adjusted = self._apply_distribution_curve(normalized)
                    micro = (row['components']['niche'] * 0.5 + row['components']['audience'] * 0.5) * 0.01
                    row['final_score'] = float(min(0.85, adjusted + micro))

            # Build final matches with explanations
            for row in scored_rows:
                score = row['final_score']
                explanation = self._generate_match_explanation(
                    query_text, abstract, industry, row['outlet'], row['outlet_id'], score
                )
                row['result'] = {
                    "outlet": row['outlet'],
                    "score": self._ensure_json_serializable(round(score, 3)),
                    "match_confidence": f"{round(score * 100)}%",
                    "match_explanation": explanation
                }

            # Sort by final_score descending and return top-N
            scored_rows.sort(key=lambda r: r['final_score'], reverse=True)
            matches = [r['result'] for r in scored_rows if r['final_score'] > self.MIN_SCORE_THRESHOLD]

            print(f"\n📊 MATCHING RESULTS:")
            print(f"   Total outlets processed: {len(outlets)}")
            print(f"   Excluded outlets: {len(excluded_outlets)}")
            print(f"   Matches found: {len(matches)}")
            if matches:
                print(f"   Score range: {matches[-1]['score']:.3f} - {matches[0]['score']:.3f}")

            return matches[:limit]

        except Exception as e:
            print(f"❌ Error in find_matches: {str(e)}")
            return []

    def _categorize_industry(self, industry: str) -> str:
        """Categorize industry for exclusion logic."""
        industry_lower = industry.lower()
        
        if any(cyber_term in industry_lower for cyber_term in ['cybersecurity', 'cyber', 'security', 'hacking']):
            return 'cybersecurity'
        elif any(edu_term in industry_lower for edu_term in ['education', 'learning', 'academic', 'teaching']):
            return 'education'
        elif any(health_term in industry_lower for health_term in ['healthcare', 'health', 'medical', 'patient']):
            return 'healthcare'
        
        return 'general'

    def _calculate_comprehensive_similarity(self, query_text: str, abstract: str, industry: str, outlet: Dict, outlet_id: str) -> float:
        """Calculate comprehensive similarity using optimized scoring."""
        try:
            # Core similarity components
            semantic_score = self._calculate_semantic_similarity(query_text, outlet_id)
            industry_score = self._calculate_industry_relevance(industry, outlet_id)
            content_score = self._calculate_content_relevance(abstract, outlet_id)
            topic_score = self._calculate_topic_similarity(query_text, outlet_id)
            
            # Enhanced niche keyword matching
            niche_score = self._calculate_niche_keyword_matching(industry, outlet_id)
            
            # Audience relevance with increased weight
            audience_score = self._calculate_audience_relevance(industry, outlet_id)
            
            # Dynamic weights based on content type
            weights = self._calculate_optimized_weights(abstract, industry)
            
            # Calculate weighted score
            weighted_score = (
                semantic_score * weights['semantic'] +
                industry_score * weights['industry'] +
                content_score * weights['content'] +
                topic_score * weights['topic'] +
                niche_score * self.NICHE_KEYWORD_WEIGHT +
                audience_score * self.AUDIENCE_WEIGHT
            )
            
            # Apply industry-specific boosting
            industry_boost = self._calculate_industry_boost(industry, outlet_id)
            
            # Apply score transformation for better differentiation
            final_score = self._apply_optimized_score_transformation(weighted_score + industry_boost)
            
            # Cap the final score to avoid 100% matches
            final_score = min(0.85, final_score)
            
            return float(final_score)
            
        except Exception as e:
            print(f"❌ Error calculating comprehensive similarity: {e}")
            return 0.0

    def _calculate_semantic_similarity(self, query_text: str, outlet_id: str) -> float:
        """Calculate semantic similarity using available NLP."""
        try:
            if self.nlp and outlet_id in self._outlet_embeddings:
                query_doc = self.nlp(query_text.lower())
                outlet_vector = self._outlet_embeddings[outlet_id]['vector']
                
                # Check if outlet vector exists and is valid
                if outlet_vector is None or not outlet_vector:
                    return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))
                
                # Convert to numpy array if it's a list
                if isinstance(outlet_vector, list):
                    outlet_vector = np.array(outlet_vector)
                
                # Validate vector dimensions
                if outlet_vector.size == 0:
                    return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))
                
                # Check if vectors have compatible dimensions
                if query_doc.vector.size != outlet_vector.size:
                    return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))
                
                # Calculate cosine similarity
                dot_product = np.dot(query_doc.vector, outlet_vector)
                query_norm = np.linalg.norm(query_doc.vector)
                outlet_norm = np.linalg.norm(outlet_vector)
                
                # Avoid division by zero
                if query_norm == 0 or outlet_norm == 0:
                    return 0.0
                
                similarity = dot_product / (query_norm * outlet_norm)
                return float(max(0.0, similarity))
            else:
                return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))
                
        except Exception as e:
            print(f"❌ Error in semantic similarity for outlet {outlet_id}: {e}")
            return self._calculate_tfidf_similarity(query_text, self._outlet_texts.get(outlet_id, ''))

    def _calculate_industry_relevance(self, industry: str, outlet_id: str) -> float:
        """Calculate industry relevance with enhanced keyword matching."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id].lower()
            industry_lower = industry.lower()
            
            # Direct keyword matching
            keyword_score = self._calculate_keyword_overlap(industry_lower, outlet_text)
            
            # Industry-specific keyword matching
            industry_keyword_score = self._calculate_industry_specific_keywords(industry_lower, outlet_text)
            
            # Semantic similarity
            semantic_score = self._calculate_tfidf_similarity(industry_lower, outlet_text)
            
            # Weighted combination
            base_score = (
                keyword_score * 0.4 + 
                industry_keyword_score * 0.4 + 
                semantic_score * 0.2
            )
            
            # Apply non-linear scaling for better differentiation
            return self._apply_industry_score_scaling(base_score)
            
        except Exception as e:
            print(f"❌ Error in industry relevance: {e}")
            return 0.0

    def _calculate_content_relevance(self, abstract: str, outlet_id: str) -> float:
        """Calculate content relevance using optimized NLP."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id]
            
            # Topic similarity
            topic_score = self._calculate_topic_similarity(abstract, outlet_id)
            
            # Keyword overlap
            keyword_score = self._calculate_keyword_overlap(abstract.lower(), outlet_text.lower())
            
            # Semantic similarity
            semantic_score = self._calculate_tfidf_similarity(abstract, outlet_text)
            
            # Technical term matching
            tech_score = self._calculate_technical_term_matching(abstract, outlet_text)
            
            # Weighted combination
            base_score = (
                topic_score * 0.3 + 
                keyword_score * 0.3 + 
                semantic_score * 0.25 + 
                tech_score * 0.15
            )
            
            return self._apply_content_score_scaling(base_score)
            
        except Exception as e:
            print(f"❌ Error in content relevance: {e}")
            return 0.0

    def _calculate_topic_similarity(self, query_text: str, outlet_id: str) -> float:
        """Calculate topic similarity with reduced threshold."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            query_topics = self._extract_topics(query_text)
            outlet_topics = self._outlet_embeddings[outlet_id].get('keywords', [])
            
            if not query_topics or not outlet_topics:
                return 0.0
            
            # Calculate Jaccard similarity
            query_set = set(query_topics)
            outlet_set = set(outlet_topics)
            
            intersection = query_set.intersection(outlet_set)
            union = query_set.union(outlet_set)
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            # Apply reduced threshold for better matching
            return similarity if similarity >= self.TOPIC_OVERLAP_THRESHOLD else similarity * 0.5
            
        except Exception as e:
            print(f"❌ Error in topic similarity: {e}")
            return 0.0

    def _calculate_niche_keyword_matching(self, industry: str, outlet_id: str) -> float:
        """Calculate niche keyword matching with enhanced weight."""
        try:
            if outlet_id not in self._outlet_keywords:
                return 0.0
            
            industry_lower = industry.lower()
            outlet_keywords = self._outlet_keywords[outlet_id]
            
            # Get industry-specific keywords
            industry_keywords = self.INDUSTRY_KEYWORDS.get(
                self._categorize_industry(industry), set()
            )
            
            if not industry_keywords:
                return 0.0
            
            # Calculate keyword matches
            matches = 0
            for outlet_keyword in outlet_keywords:
                if outlet_keyword in industry_keywords:
                    matches += 1
            
            # Calculate score based on match ratio
            score = matches / len(industry_keywords) if industry_keywords else 0.0
            
            # Apply enhanced weighting for niche keywords
            return score * self.NICHE_KEYWORD_WEIGHT
            
        except Exception as e:
            print(f"❌ Error in niche keyword matching: {e}")
            return 0.0

    def _calculate_audience_relevance(self, industry: str, outlet_id: str) -> float:
        """Calculate audience relevance with increased weight."""
        try:
            if outlet_id not in self._outlet_audiences:
                return 0.0
            
            industry_lower = industry.lower()
            outlet_audiences = self._outlet_audiences[outlet_id]
            
            # Calculate audience overlap
            matches = 0
            for audience in outlet_audiences:
                if any(term in audience for term in industry_lower.split()):
                    matches += 1
            
            # Calculate score
            score = matches / len(outlet_audiences) if outlet_audiences else 0.0
            
            # Apply audience weight boost
            return score * self.AUDIENCE_WEIGHT
            
        except Exception as e:
            print(f"❌ Error in audience relevance: {e}")
            return 0.0

    def _calculate_optimized_weights(self, abstract: str, industry: str) -> Dict[str, float]:
        """Calculate optimized weights based on content characteristics."""
        abstract_lower = abstract.lower()
        industry_lower = industry.lower()
        
        # More conservative base weights
        weights = {
            'semantic': 0.20,
            'industry': 0.25,
            'content': 0.20,
            'topic': 0.25,
            'niche': 0.10
        }
        
        # Adjust based on content characteristics
        if any(tech_term in abstract_lower for tech_term in ['ai', 'artificial intelligence', 'machine learning', 'technology']):
            weights['content'] += 0.03
            weights['topic'] += 0.03
            weights['semantic'] -= 0.03
            weights['industry'] -= 0.03
        
        # Industry-specific adjustments (more conservative)
        industry_category = self._categorize_industry(industry)
        if industry_category in ['education', 'healthcare', 'cybersecurity']:
            weights['industry'] += 0.05  # Reduced from 0.10
            weights['niche'] += 0.03     # Reduced from 0.05
            weights['content'] -= 0.03   # Reduced from 0.05
            weights['topic'] -= 0.03     # Reduced from 0.05
            weights['semantic'] -= 0.02  # Reduced from 0.05
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        return weights

    def _apply_optimized_score_transformation(self, score: float) -> float:
        """Apply optimized score transformation for more realistic differentiation."""
        # More conservative scoring to avoid 100% matches
        if score > 0.85:
            result = 0.75 + (score - 0.85) * 0.5  # 75-80% for excellent matches
        elif score > 0.7:
            result = 0.65 + (score - 0.7) * 0.67  # 65-75% for very good matches
        elif score > 0.5:
            result = 0.50 + (score - 0.5) * 0.75  # 50-65% for good matches
        elif score > 0.3:
            result = 0.35 + (score - 0.3) * 0.75  # 35-50% for moderate matches
        elif score > 0.15:
            result = 0.20 + (score - 0.15) * 1.0  # 20-35% for weak matches
        else:
            result = score * 1.33  # 0-20% for poor matches
        
        return float(result)

    def _apply_industry_score_scaling(self, score: float) -> float:
        """Apply industry-specific score scaling with more conservative ranges."""
        if score > 0.8:
            return 0.70 + (score - 0.8) * 0.5  # 70-75% for excellent industry matches
        elif score > 0.6:
            return 0.60 + (score - 0.6) * 0.5  # 60-70% for very good matches
        elif score > 0.4:
            return 0.45 + (score - 0.4) * 0.75  # 45-60% for good matches
        elif score > 0.2:
            return 0.30 + (score - 0.2) * 0.75  # 30-45% for moderate matches
        elif score > 0.1:
            return 0.20 + (score - 0.1) * 1.0  # 20-30% for weak matches
        else:
            return score * 2.0  # 0-20% for poor matches

    def _apply_content_score_scaling(self, score: float) -> float:
        """Apply content-specific score scaling with more conservative ranges."""
        if score > 0.8:
            return 0.70 + (score - 0.8) * 0.5  # 70-75% for excellent content matches
        elif score > 0.6:
            return 0.55 + (score - 0.6) * 0.75  # 55-70% for very good matches
        elif score > 0.4:
            return 0.40 + (score - 0.4) * 0.75  # 40-55% for good matches
        elif score > 0.2:
            return 0.25 + (score - 0.2) * 0.75  # 25-40% for moderate matches
        else:
            return score * 1.25  # 0-25% for weak matches

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using available NLP."""
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
            print(f"❌ Error extracting topics: {e}")
            return []

    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap between two texts."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
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
                'technology', 'tech', 'software', 'ai', 'artificial intelligence',
                'cybersecurity', 'security', 'cyber', 'hacking', 'malware'
            }
            
            important_matches = intersection.intersection(important_terms)
            if important_matches:
                boost = len(important_matches) * 0.1
                base_score = min(1.0, base_score + boost)
            
            return base_score
            
        except Exception as e:
            print(f"❌ Error in keyword overlap: {e}")
            return 0.0

    def _calculate_industry_specific_keywords(self, industry: str, outlet_text: str) -> float:
        """Calculate industry-specific keyword matching."""
        try:
            industry_lower = industry.lower()
            industry_category = self._categorize_industry(industry)
            
            # Get industry keywords
            industry_keywords = self.INDUSTRY_KEYWORDS.get(industry_category, set())
            
            if not industry_keywords:
                return self._calculate_keyword_overlap(industry_lower, outlet_text)
            
            # Calculate keyword matching
            matching_keywords = [keyword for keyword in industry_keywords if keyword in outlet_text]
            
            if matching_keywords:
                total_matches = sum(outlet_text.count(keyword) for keyword in matching_keywords)
                score = min(1.0, total_matches / (len(outlet_text.split()) * 0.1))
                return score
            
            return 0.0
            
        except Exception as e:
            print(f"❌ Error in industry-specific keyword matching: {e}")
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
            print(f"❌ Error in technical term matching: {e}")
            return 0.0

    def _calculate_industry_boost(self, industry: str, outlet_id: str) -> float:
        """Calculate industry-specific boost for excellent matches."""
        try:
            if outlet_id not in self._outlet_embeddings:
                return 0.0
            
            outlet_text = self._outlet_texts[outlet_id].lower()
            industry_lower = industry.lower()
            industry_category = self._categorize_industry(industry)
            
            # Get high-value keywords for the industry
            high_value_keywords = self.INDUSTRY_KEYWORDS.get(industry_category, set())
            
            if not high_value_keywords:
                return 0.0
            
            # Calculate boost based on high-value keyword matches
            matching_keywords = [keyword for keyword in high_value_keywords if keyword in outlet_text]
            
            if matching_keywords:
                total_matches = sum(outlet_text.count(keyword) for keyword in matching_keywords)
                boost = min(0.25, total_matches * 0.03)
                return boost
            
            return 0.0
            
        except Exception as e:
            print(f"❌ Error in industry boost calculation: {e}")
            return 0.0

    def _generate_match_explanation(self, query_text: str, abstract: str, industry: str, outlet: Dict, outlet_id: str, score: float) -> List[str]:
        """Generate detailed match explanation."""
        try:
            explanation = []
            
            # Add score-based explanation with updated ranges
            if score > 0.75:
                explanation.append("Excellent match with high relevance across all dimensions")
            elif score > 0.65:
                explanation.append("Strong match with good alignment to content and industry")
            elif score > 0.50:
                explanation.append("Good match with moderate relevance indicators")
            elif score > 0.35:
                explanation.append("Moderate match with some relevant connections")
            elif score > 0.20:
                explanation.append("Weak match with limited relevance")
            else:
                explanation.append("Poor match with minimal relevance detected")
            
            # Add specific matching details
            if outlet_id in self._outlet_embeddings:
                # Industry matching details
                industry_score = self._calculate_industry_relevance(industry, outlet_id)
                if industry_score > 0.3:
                    explanation.append(f"Industry alignment: {round(industry_score * 100)}%")
                
                # Content matching details
                content_score = self._calculate_content_relevance(abstract, outlet_id)
                if content_score > 0.3:
                    explanation.append(f"Content relevance: {round(content_score * 100)}%")
                
                # Topic matching details
                topic_score = self._calculate_topic_similarity(query_text, outlet_id)
                if topic_score > 0.2:
                    explanation.append(f"Topic similarity: {round(topic_score * 100)}%")
            
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
            print(f"❌ Error generating explanation: {e}")
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
            
            return float(similarity)
        except Exception as e:
            print(f"❌ Error in TF-IDF similarity: {e}")
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
            print(f"❌ Error fetching outlets: {str(e)}")
            return []

   