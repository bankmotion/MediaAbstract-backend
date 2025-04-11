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
        # Load spaCy model for advanced NLP
        self.nlp = spacy.load("en_core_web_md")
        
        # Initialize TF-IDF vectorizer with adjusted parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0  # Adjusted from 0.95 to 1.0 to handle small document sets
        )
        
        self.supabase = supabase_client
        self.threshold = 0.3
        
        # Initialize data structures for comprehensive learning
        self.learned_data = {
            'keywords': set(),
            'audiences': set(),
            'sections': set(),
            'guidelines': {},
            'pitch_tips': set(),
            'topics': set(),
            'content_requirements': {},
            'outlet_specializations': {}
        }
        
        # Dynamic weights that will be adjusted based on data patterns
        self.field_weights = {
            'Industry Match': 3.5,      # Highest priority - correct industry targeting
            'Keywords': 3.0,            # Direct topic relevance
            'Audience': 2.8,            # Target audience match
            'Content Type': 2.5,        # Section and format match
            'Requirements': 2.0,        # Guidelines and pitch tips
            'Outlet Expertise': 2.0,    # Outlet's specialized areas
            'Prestige': 1.5            # Publication quality
        }
        
        # Actual prestige levels from data
        self.prestige_levels = {
            'High': 1.0,
            'Medium': 0.8,
            'Low': 0.6
        }

        # Content type mapping based on Section Names
        self.content_types = {
            'Opinion': ['opinion', 'perspective', 'viewpoint', 'commentary'],
            'Guest Post': ['guest post', 'contributed content', 'submission'],
            'Features': ['feature', 'in-depth', 'analysis'],
            'News': ['news', 'current events', 'breaking'],
            'Insights': ['insight', 'research', 'analysis'],
            'Technology': ['tech', 'technology', 'software', 'digital'],
            'Commentary': ['commentary', 'opinion', 'analysis']
        }

        # Industry categories from actual data
        self.industry_categories = {
            'Tech/Software': {
                'keywords': ['software development', 'programming', 'DevOps', 'tech', 'IT', 'cybersecurity'],
                'outlets': ['Dark Reading', 'TechCrunch', 'The Next Web', 'InfoWorld', 'ZDNet']
            },
            'Business/Finance': {
                'keywords': ['finance', 'business', 'economics', 'corporate strategy', 'leadership'],
                'outlets': ['Fortune', 'The Wall Street Journal', 'Bloomberg', 'Harvard Business Review']
            },
            'Marketing': {
                'keywords': ['marketing', 'advertising', 'media', 'content strategy', 'digital marketing'],
                'outlets': ['MarketingProfs', 'AdWeek', 'Content Marketing Institute']
            },
            'Healthcare': {
                'keywords': ['healthcare', 'medical', 'health tech', 'patient care'],
                'outlets': ['Healthcare IT News', 'Modern Healthcare', 'MedCityNews']
            },
            'Sustainability': {
                'keywords': ['sustainability', 'environmental', 'green tech', 'climate'],
                'outlets': ['GreenBiz', 'Environmental Leader']
            },
            'Cybersecurity': {
                'keywords': ['cybersecurity', 'security', 'cyber threats', 'information security'],
                'outlets': ['Dark Reading', 'SecurityWeek', 'CSO Online']
            }
        }
        
        # Load and analyze outlet data
        self._initialize_matcher()
        
        # Load user feedback data
        self._load_feedback_data()

    def _initialize_matcher(self):
        """Initialize and prepare the matcher with comprehensive data analysis."""
        try:
            # Load all outlets
            outlets = self.get_outlets()
            if not outlets:
                return

            # Analyze and categorize outlets
            self._analyze_outlets(outlets)
            
            # Build specialized matching patterns
            self._build_matching_patterns()
            
            # Calculate outlet expertise scores
            self._calculate_outlet_expertise()
            
        except Exception as e:
            print(f"Error initializing matcher: {str(e)}")

    def _analyze_outlets(self, outlets: List[Dict]):
        """Comprehensive analysis of outlet data."""
        for outlet in outlets:
            # Process Keywords and Topics
            if outlet.get('Keywords'):
                keywords = [k.strip().lower() for k in outlet['Keywords'].split(',')]
                self.learned_data['keywords'].update(keywords)
                
                # Extract topics and themes
                doc = self.nlp(' '.join(keywords))
                topics = [token.text for token in doc if not token.is_stop]
                self.learned_data['topics'].update(topics)

            # Analyze Audience Patterns
            if outlet.get('Audience'):
                audience = outlet.get('Audience').lower()
                self.learned_data['audiences'].add(audience)
                
                # Build audience specialization mapping
                outlet_name = outlet.get('Outlet Name')
                if outlet_name:
                    self.learned_data['outlet_specializations'][outlet_name] = {
                        'audience': audience,
                        'topics': keywords if outlet.get('Keywords') else [],
                        'prestige': outlet.get('Prestige', 'Medium'),
                        'section': outlet.get('Section Name', '')
                    }

            # Process Content Requirements
            if outlet.get('Guidelines'):
                self._extract_content_requirements(outlet)

            # Analyze Pitch Tips for patterns
            if outlet.get('Pitch Tips'):
                self._analyze_pitch_tips(outlet)

    def _extract_content_requirements(self, outlet: Dict):
        """Extract detailed content requirements from guidelines."""
        guidelines = outlet.get('Guidelines', '').lower()
        outlet_name = outlet.get('Outlet Name')
        
        requirements = {
            'word_count': None,
            'format': None,
            'style': None,
            'special_requirements': []
        }

        # Extract word count
        words = guidelines.split()
        for i, word in enumerate(words):
            if word.isdigit() and i > 0:
                if words[i-1].lower() in ['words', 'word']:
                    requirements['word_count'] = int(word)

        # Identify format requirements
        format_indicators = ['article', 'opinion piece', 'guest post', 'feature']
        for indicator in format_indicators:
            if indicator in guidelines:
                requirements['format'] = indicator
                break

        # Extract style preferences
        style_indicators = {
            'technical': ['technical', 'detailed', 'in-depth'],
            'conversational': ['conversational', 'engaging', 'accessible'],
            'professional': ['professional', 'formal', 'business-style']
        }
        
        for style, indicators in style_indicators.items():
            if any(ind in guidelines for ind in indicators):
                requirements['style'] = style
                break

        self.learned_data['content_requirements'][outlet_name] = requirements

    def _analyze_pitch_tips(self, outlet: Dict):
        """Analyze pitch tips for detailed requirements and preferences."""
        pitch_tips = outlet.get('Pitch Tips', '').lower()
        outlet_name = outlet.get('Outlet Name')
        
        preferences = {
            'focus_areas': [],
            'avoid': [],
            'requirements': [],
            'style_preferences': []
        }

        # Extract focus areas
        if 'focus on' in pitch_tips:
            focus_start = pitch_tips.index('focus on') + 8
            focus_end = pitch_tips.find('.', focus_start)
            if focus_end != -1:
                focus_area = pitch_tips[focus_start:focus_end].strip()
                preferences['focus_areas'].append(focus_area)

        # Extract things to avoid
        if 'avoid' in pitch_tips:
            avoid_start = pitch_tips.index('avoid') + 5
            avoid_end = pitch_tips.find('.', avoid_start)
            if avoid_end != -1:
                avoid_item = pitch_tips[avoid_start:avoid_end].strip()
                preferences['avoid'].append(avoid_item)

        # Extract specific requirements
        requirement_indicators = ['must', 'should', 'require', 'need']
        for indicator in requirement_indicators:
            if indicator in pitch_tips:
                idx = pitch_tips.index(indicator)
                end_idx = pitch_tips.find('.', idx)
                if end_idx != -1:
                    req = pitch_tips[idx:end_idx].strip()
                    preferences['requirements'].append(req)

        self.learned_data['pitch_tips'] = preferences

    def calculate_similarity_score(self, outlet: Dict, query: str, industry: str) -> float:
        """Calculate comprehensive similarity score using advanced matching."""
        try:
            scores = {}
            
            # 1. Industry Match Score
            industry_score = self._calculate_industry_match(outlet, industry)
            scores['Industry Match'] = industry_score * self.field_weights['Industry Match']
            
            # 2. Keyword and Topic Relevance
            keyword_score = self._calculate_keyword_relevance(outlet, query)
            scores['Keywords'] = keyword_score * self.field_weights['Keywords']
            
            # 3. Audience Match
            audience_score = self._calculate_audience_match(outlet, query, industry)
            scores['Audience'] = audience_score * self.field_weights['Audience']
            
            # 4. Content Type Compatibility
            content_score = self._calculate_content_compatibility(outlet, query)
            scores['Content Type'] = content_score * self.field_weights['Content Type']
            
            # 5. Requirements Match
            req_score = self._calculate_requirements_match(outlet, query)
            scores['Requirements'] = req_score * self.field_weights['Requirements']
            
            # 6. Outlet Expertise Score
            expertise_score = self._calculate_outlet_expertise_match(outlet, query, industry)
            scores['Outlet Expertise'] = expertise_score * self.field_weights['Outlet Expertise']
            
            # 7. Prestige Factor
            prestige_score = self._calculate_prestige_score(outlet)
            scores['Prestige'] = prestige_score * self.field_weights['Prestige']
            
            # Calculate weighted average
            total_weight = sum(self.field_weights.values())
            final_score = sum(scores.values()) / total_weight
            
            # Apply bonus for exceptional matches
            if self._check_exceptional_match(outlet, query, industry):
                final_score = min(1.0, final_score * 1.2)
            
            return final_score
            
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def _calculate_industry_match(self, outlet: Dict, industry: str) -> float:
        """Calculate industry match score with advanced matching."""
        try:
            # Get outlet's industry indicators
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_audience = outlet.get('Audience', '').lower()
            outlet_section = outlet.get('Section Name', '').lower()
            
            # Industry match components
            keyword_match = any(kw in outlet_keywords for kw in industry.lower().split())
            audience_match = industry.lower() in outlet_audience
            section_match = industry.lower() in outlet_section
            
            # Calculate weighted score
            score = 0.0
            if keyword_match: score += 0.5
            if audience_match: score += 0.3
            if section_match: score += 0.2
            
            # Boost score for specialized outlets
            if outlet.get('Outlet Name') in self.learned_data['outlet_specializations']:
                specialization = self.learned_data['outlet_specializations'][outlet.get('Outlet Name')]
                if industry.lower() in ' '.join(specialization['topics']).lower():
                    score = min(1.0, score * 1.3)
            
            return score
            
        except Exception as e:
            print(f"Error calculating industry match: {str(e)}")
            return 0.0

    def _calculate_keyword_relevance(self, outlet: Dict, query: str) -> float:
        """Calculate keyword relevance with semantic analysis."""
        try:
            # Process query and outlet keywords
            query_doc = self.nlp(query.lower())
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_doc = self.nlp(outlet_keywords)
            
            # Calculate semantic similarity
            semantic_score = query_doc.similarity(outlet_doc)
            
            # Calculate keyword overlap
            query_words = set(query.lower().split())
            outlet_words = set(outlet_keywords.split(','))
            overlap_score = len(query_words & outlet_words) / len(query_words | outlet_words) if outlet_words else 0
            
            # Combine scores with weights
            final_score = (semantic_score * 0.7) + (overlap_score * 0.3)
            
            return final_score
            
        except Exception as e:
            print(f"Error calculating keyword relevance: {str(e)}")
            return 0.0

    def _calculate_audience_match(self, outlet: Dict, query: str, industry: str) -> float:
        """Calculate audience match score with advanced matching."""
        try:
            # Get outlet's audience indicators
            outlet_audience = outlet.get('Audience', '').lower()
            
            # Audience match components
            audience_match = industry.lower() in outlet_audience
            
            # Calculate weighted score
            score = 0.0
            if audience_match: score = 0.8
            
            return score
            
        except Exception as e:
            print(f"Error calculating audience match: {str(e)}")
            return 0.0

    def _calculate_content_compatibility(self, outlet: Dict, query: str) -> float:
        """Calculate content compatibility score with advanced matching."""
        try:
            # Get outlet's content type indicators
            outlet_section = outlet.get('Section Name', '').lower()
            
            # Content type match components
            section_match = outlet_section in self.learned_data['sections']
            
            # Calculate weighted score
            score = 0.0
            if section_match: score = 0.7
            
            return score
            
        except Exception as e:
            print(f"Error calculating content compatibility: {str(e)}")
            return 0.0

    def _calculate_requirements_match(self, outlet: Dict, query: str) -> float:
        """Calculate requirements match score with advanced matching."""
        try:
            # Get outlet's requirements indicators
            outlet_name = outlet.get('Outlet Name')
            if outlet_name:
                requirements = self.learned_data['content_requirements'].get(outlet_name, {})
                
                # Requirements match components
                word_count_match = requirements.get('word_count') and requirements['word_count'] == len(query.split())
                format_match = requirements.get('format') and requirements['format'] in query.lower()
                style_match = requirements.get('style') and requirements['style'] in query.lower()
                
                # Calculate weighted score
                score = 0.0
                if word_count_match: score += 0.5
                if format_match: score += 0.3
                if style_match: score += 0.2
                
                return score
            
        except Exception as e:
            print(f"Error calculating requirements match: {str(e)}")
            return 0.0

    def _calculate_outlet_expertise_match(self, outlet: Dict, query: str, industry: str) -> float:
        """Calculate outlet expertise match score with advanced matching."""
        try:
            # Get outlet's expertise indicators
            outlet_name = outlet.get('Outlet Name')
            if outlet_name:
                specialization = self.learned_data['outlet_specializations'].get(outlet_name, {})
                
                # Expertise match components
                topic_match = any(topic in query.lower() for topic in specialization.get('topics', []))
                audience_match = specialization.get('audience') and specialization['audience'] in query.lower()
                section_match = specialization.get('section') and specialization['section'] in query.lower()
                
                # Calculate weighted score
                score = 0.0
                if topic_match: score += 0.5
                if audience_match: score += 0.3
                if section_match: score += 0.2
                
                return score
            
        except Exception as e:
            print(f"Error calculating outlet expertise match: {str(e)}")
            return 0.0

    def _calculate_prestige_score(self, outlet: Dict) -> float:
        """Calculate prestige score with advanced matching."""
        try:
            # Get outlet's prestige indicators
            outlet_name = outlet.get('Outlet Name')
            if outlet_name:
                prestige = outlet.get('Prestige', 'Medium')
                prestige_level = self.prestige_levels.get(prestige, 0.7)
                
                return prestige_level
            
        except Exception as e:
            print(f"Error calculating prestige score: {str(e)}")
            return 0.0

    def _check_exceptional_match(self, outlet: Dict, query: str, industry: str) -> bool:
        """Check if the match is exceptional with advanced matching."""
        try:
            # Get outlet's exceptional indicators
            outlet_name = outlet.get('Outlet Name')
            if outlet_name:
                exceptional_indicators = {
                    'exceptional_keywords': ['exceptional', 'unique', 'rare', 'rare opportunity'],
                    'exceptional_audiences': ['exceptional', 'unique', 'rare', 'rare opportunity'],
                    'exceptional_sections': ['exceptional', 'unique', 'rare', 'rare opportunity'],
                    'exceptional_guidelines': ['exceptional', 'unique', 'rare', 'rare opportunity'],
                    'exceptional_pitch_tips': ['exceptional', 'unique', 'rare', 'rare opportunity']
                }
                
                # Check exceptional indicators
                for field, indicators in exceptional_indicators.items():
                    if any(ind in query.lower() for ind in indicators):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error checking exceptional match: {str(e)}")
            return False

    def _generate_match_explanation(self, outlet: Dict, score: float, query: str, industry: str) -> str:
        """Generate detailed match explanation with specific insights."""
        try:
            reasons = []
            
            # Outlet Quality and Prestige
            prestige = outlet.get('Prestige', '')
            if prestige:
                reasons.append(f"{prestige} prestige publication")
            
            # Industry Match
            industry_match = self._calculate_industry_match(outlet, industry)
            if industry_match > 0.8:
                reasons.append(f"Strong {industry} industry focus")
            elif industry_match > 0.5:
                reasons.append(f"Relevant to {industry} sector")
            
            # Keyword Matches
            keywords = outlet.get('Keywords', '').lower().split(',')
            matching_keywords = [kw.strip() for kw in keywords if kw.strip() in query.lower()]
            if matching_keywords:
                reasons.append(f"Matching topics: {', '.join(matching_keywords)}")
            
            # Audience Match
            audience = outlet.get('Audience', '')
            if audience:
                reasons.append(f"Target audience: {audience}")
            
            # Content Requirements
            if outlet.get('Outlet Name') in self.learned_data['content_requirements']:
                reqs = self.learned_data['content_requirements'][outlet.get('Outlet Name')]
                if reqs['word_count']:
                    reasons.append(f"Word count requirement: {reqs['word_count']} words")
                if reqs['format']:
                    reasons.append(f"Preferred format: {reqs['format']}")
                if reqs['style']:
                    reasons.append(f"Writing style: {reqs['style']}")
            
            # Match Quality Indicator
            if score >= 0.8:
                reasons.append("Excellent match for your content")
            elif score >= 0.6:
                reasons.append("Strong potential match")
            elif score >= 0.4:
                reasons.append("Good match with specific focus required")
            else:
                reasons.append("Potential match with careful targeting needed")
        
            return "; ".join(reasons)
            
        except Exception as e:
            print(f"Error generating match explanation: {str(e)}")
            return "Match explanation unavailable"

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

    def _calculate_outlet_expertise(self):
        """Calculate expertise scores for outlets based on their specialization."""
        try:
            for outlet_name, specialization in self.learned_data['outlet_specializations'].items():
                expertise_score = 0.0
                
                # Score based on topic diversity
                topics = specialization.get('topics', [])
                if topics:
                    expertise_score += min(1.0, len(topics) / 10)  # Cap at 10 topics
                
                # Score based on audience specificity
                audience = specialization.get('audience', '')
                if audience:
                    audience_doc = self.nlp(audience)
                    specificity_terms = [token.text for token in audience_doc if not token.is_stop]
                    expertise_score += min(1.0, len(specificity_terms) / 5)  # Cap at 5 terms
                
                # Score based on prestige
                prestige = specialization.get('prestige', 'Medium')
                prestige_score = self.prestige_levels.get(prestige, 0.7)
                expertise_score += prestige_score
                
                # Normalize final score
                expertise_score = min(1.0, expertise_score / 3)  # Average of the three components
                
                # Store the expertise score
                self.learned_data['outlet_specializations'][outlet_name]['expertise_score'] = expertise_score
                
        except Exception as e:
            print(f"Error calculating outlet expertise: {str(e)}")

    def _build_matching_patterns(self):
        """Build specialized matching patterns from learned data."""
        try:
            # Build keyword patterns
            self.keyword_patterns = {}
            for outlet_name, specialization in self.learned_data['outlet_specializations'].items():
                topics = specialization.get('topics', [])
                if topics:
                    # Create TF-IDF vectors for topics
                    topic_text = ' '.join(topics)
                    # Handle single document case
                    if len(topics) == 1:
                        self.keyword_patterns[outlet_name] = {
                            'topics': topics,
                            'text': topic_text  # Store raw text instead of vector for single documents
                        }
                    else:
                        self.keyword_patterns[outlet_name] = {
                            'topics': topics,
                            'vector': self.vectorizer.fit_transform([topic_text])
                        }

            # Build audience patterns
            self.audience_patterns = {}
            for audience in self.learned_data['audiences']:
                # Extract key terms from audience descriptions
                audience_doc = self.nlp(audience)
                key_terms = [token.text for token in audience_doc if not token.is_stop]
                self.audience_patterns[audience] = key_terms

            # Build section patterns
            self.section_patterns = {}
            for section in self.learned_data['sections']:
                # Map sections to content types
                matched_types = []
                for content_type, indicators in self.content_types.items():
                    if any(ind in section.lower() for ind in indicators):
                        matched_types.append(content_type)
                self.section_patterns[section] = matched_types

        except Exception as e:
            print(f"Error building matching patterns: {str(e)}")

    def find_matches(self, query: str, industry: str, limit: int = 20) -> List[Dict]:
        """Find matches with comprehensive scoring and ranking."""
        try:
            # Get all outlets
            outlets = self.get_outlets()
            if not outlets:
                print("No outlets found in database")
                return []

            # Calculate scores and prepare matches
            matches = []
            for outlet in outlets:
                # Calculate comprehensive similarity score
                score = self.calculate_similarity_score(outlet, query, industry)
                
                # Only include matches above threshold
                if score >= self.threshold:
                    matches.append({
                        "outlet": outlet,
                        "score": score,
                        "match_confidence": 0.0,  # Will be normalized
                        "match_explanation": self._generate_match_explanation(outlet, score, query, industry)
                    })

            # Normalize scores
            if matches:
                max_score = max(m['score'] for m in matches)
                min_score = min(m['score'] for m in matches)
                score_range = max_score - min_score
                
                for match in matches:
                    if score_range > 0:
                        # Normalize to 0-100 range with better distribution
                        normalized_score = (match['score'] - min_score) / score_range
                        match['match_confidence'] = round(normalized_score * 100, 2)
                    else:
                        # If all scores are the same, assign based on threshold
                        match['match_confidence'] = round((match['score'] / self.threshold) * 100, 2)
                    
                    # Round the raw score for cleaner display
                    match['score'] = round(match['score'], 4)

            # Sort by score (descending) and apply limit
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:limit]

        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            return []