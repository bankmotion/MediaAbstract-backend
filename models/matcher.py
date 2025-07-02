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
        # Load spaCy model for advanced NLP with error handling
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            # Fallback to smaller model if medium model not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If no spaCy model available, use a simple text processor
                print("Warning: spaCy model not available, using fallback text processing")
                self.nlp = None
        
        # Initialize TF-IDF vectorizer with adjusted parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0
        )
        
        self.supabase = supabase_client
        self.threshold = 0.15  # Base threshold for general content
        
        # Initialize data structures for comprehensive learning
        self.learned_data = {
            'keywords': set(),
            'audiences': set(),
            'sections': set(),
            'guidelines': {},
            'pitch_tips': set(),
            'topics': set(),
            'content_requirements': {},
            'outlet_specializations': {},
            'recent_articles': {}
        }
        
        # Optimized field weights for wider score range
        self.field_weights = {
            'Industry Match': 5.0,      # Increased for stronger industry influence
            'Keywords': 4.5,            # Increased for stronger keyword influence
            'News Match': 3.5,          # Increased for stronger news influence
            'Audience': 2.0,            # Reduced - less important than editorial focus
            'Content Type': 1.5,        # Reduced - less important than editorial focus
            'Requirements': 1.5,        # Reduced - less important than editorial focus
            'Outlet Expertise': 3.0,    # Increased for stronger expertise influence
            'Prestige': 1.0            # Reduced - less important than editorial focus
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
                print("No outlets found during initialization")
                return

            print(f"Initializing matcher with {len(outlets)} outlets")

            # Analyze and categorize outlets
            self._analyze_outlets(outlets)
            
            # Build specialized matching patterns
            self._build_matching_patterns()
            
            # Calculate outlet expertise scores
            self._calculate_outlet_expertise()
            
            print(f"Matcher initialization complete. Learned data: {len(self.learned_data['keywords'])} keywords, {len(self.learned_data['outlet_specializations'])} outlet specializations")
            
        except Exception as e:
            print(f"Error initializing matcher: {str(e)}")
            import traceback
            traceback.print_exc()

    def _analyze_outlets(self, outlets: List[Dict]):
        """Comprehensive analysis of outlet data."""
        for outlet in outlets:
            # Process Keywords and Topics
            if outlet.get('Keywords'):
                keywords = [k.strip().lower() for k in outlet['Keywords'].split(',')]
                self.learned_data['keywords'].update(keywords)
                
                # Extract topics and themes
                if self.nlp is not None:
                    doc = self.nlp(' '.join(keywords))
                    topics = [token.text for token in doc if not token.is_stop]
                    self.learned_data['topics'].update(topics)
                else:
                    # Fallback when spaCy is not available
                    self.learned_data['topics'].update(keywords)

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
        """Calculate simplified similarity score for faster processing."""
        try:
            # Simplified scoring - focus on basic keyword matching
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_audience = outlet.get('Audience', '').lower()
            outlet_name = outlet.get('Outlet Name', '').lower()
            
            query_lower = query.lower()
            industry_lower = industry.lower()
            
            # Basic keyword matching
            score = 0.0
            
            # Check outlet keywords
            if outlet_keywords:
                keywords_list = [kw.strip() for kw in outlet_keywords.split(',')]
                for keyword in keywords_list:
                    if keyword in query_lower or keyword in industry_lower:
                        score += 0.3
                    # Check for partial matches
                    elif any(word in keyword for word in query_lower.split()):
                        score += 0.1
            
            # Check outlet name
            if outlet_name:
                if any(word in outlet_name for word in query_lower.split()):
                    score += 0.2
                if any(word in outlet_name for word in industry_lower.split()):
                    score += 0.2
            
            # Check audience
            if outlet_audience:
                if any(word in outlet_audience for word in query_lower.split()):
                    score += 0.1
                if any(word in outlet_audience for word in industry_lower.split()):
                    score += 0.1
            
            # Basic prestige boost
            prestige = outlet.get('Prestige', 'Medium')
            if prestige == 'High':
                score += 0.1
            elif prestige == 'Medium':
                score += 0.05
            
            # Cap the score
            return min(1.0, score)
            
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
            
            # Detect specialized content across all categories
            content_specialization = self._detect_content_specialization("", industry)  # Pass empty query, check industry only
            
            # Determine outlet focus
            outlet_focus = self._determine_outlet_focus(outlet)
            
            # Industry match components
            keyword_match = any(kw in outlet_keywords for kw in industry.lower().split())
            audience_match = industry.lower() in outlet_audience
            section_match = industry.lower() in outlet_section
            
            # Calculate weighted score with improved matching
            score = 0.0
            if keyword_match: score += 0.7  # Increased from 0.6
            if audience_match: score += 0.3
            if section_match: score += 0.2
            
            # Apply specialization-aware adjustments with CLIENT'S SPECIFIC REQUIREMENTS
            if content_specialization:
                specialization_type = content_specialization['type']
                
                # Define focus compatibility matrix - CLIENT'S STRICT REQUIREMENTS
                focus_compatibility = {
                    'cybersecurity': ['cybersecurity'],  # Only cybersecurity specialists
                    'ai_ml': ['ai_ml'],                  # Only AI specialists
                    'fintech': ['fintech', 'finance'],    # Fintech or finance specialists only
                    'startup': ['startup', 'business', 'investment'],
                    'investment': ['investment', 'business', 'finance'],
                    'marketing': ['marketing'],           # Only marketing specialists
                    'healthcare': ['healthcare'],         # Only healthcare specialists
                    'education': ['education'],           # Only education specialists
                    'sustainability': ['sustainability'], # Only sustainability specialists
                    'real_estate': ['real_estate'],       # Only real estate specialists
                    'leadership': ['leadership', 'management', 'executive', 'ceo', 'c-suite', 'corporate strategy', 'organizational development', 'change management', 'talent management', 'workplace culture', 'diversity inclusion', 'remote work'],
                }
                
                # Check if outlet focus is compatible with content specialization
                compatible_focuses = focus_compatibility.get(specialization_type, [])
                
                if outlet_focus in compatible_focuses:
                    if outlet_focus == specialization_type:
                        # Perfect match - specialist outlet for specialist content
                        score = min(1.0, score * 2.0)  # Reduced from 2.5 - more reasonable bonus
                    elif outlet_focus == 'tech_general' and specialization_type in ['cybersecurity', 'ai_ml']:
                        # Tech outlet for tech-specialized content - moderate score
                        score = score * 0.6  # Increased from 0.4 - less strict
                    elif outlet_focus == 'business' and specialization_type in ['startup', 'investment', 'leadership']:
                        # Business outlet for business-specialized content
                        score = score * 0.7  # Increased from 0.5 - less strict
                    elif outlet_focus == 'finance' and specialization_type == 'fintech':
                        # Finance outlet for fintech content
                        score = score * 0.7  # Increased from 0.5 - less strict
                    else:
                        # Compatible but not perfect match - moderate score
                        score = score * 0.5  # Increased from 0.2 - less strict
                else:
                    # Incompatible focus - REDUCED PENALTY
                    score = score * 0.1  # Increased from 0.01 - less aggressive exclusion
                
                # CLIENT'S REQUIREMENT: Downrank/exclude irrelevant verticals - REDUCED PENALTY
                irrelevant_verticals = {
                    'cybersecurity': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'ai_ml': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'fintech': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'startup': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'investment': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'marketing': ['construction', 'healthcare', 'banking', 'supply chain', 'real estate'],
                    'healthcare': ['construction', 'banking', 'supply chain', 'retail', 'real estate'],
                    'education': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'sustainability': ['construction', 'healthcare', 'banking', 'supply chain', 'retail'],
                    'real_estate': ['construction', 'healthcare', 'banking', 'supply chain', 'retail'],
                    'leadership': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate']
                }
                
                irrelevant_terms = irrelevant_verticals.get(specialization_type, [])
                if any(term in outlet_keywords or term in outlet_audience for term in irrelevant_terms):
                    score = score * 0.3  # Increased from 0.1 - less aggressive penalty
                
                # Additional penalty for generic terms in specialized content - REDUCED PENALTY
                generic_terms = {
                    'cybersecurity': ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'ai_ml': ['technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'fintech': ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'startup': ['business', 'technology', 'digital', 'marketing', 'advertising'],
                    'investment': ['business', 'finance', 'technology', 'marketing', 'advertising'],
                    'marketing': ['business', 'technology', 'digital'],
                    'healthcare': ['technology', 'digital', 'software', 'marketing', 'advertising'],
                    'education': ['technology', 'digital', 'software', 'marketing', 'advertising'],
                    'sustainability': ['technology', 'digital', 'business', 'marketing', 'advertising'],
                    'real_estate': ['technology', 'digital', 'business', 'marketing', 'advertising'],
                    'leadership': ['business', 'technology', 'marketing', 'advertising']
                }
                
                generic_penalty_terms = generic_terms.get(specialization_type, [])
                generic_matches = sum(1 for term in generic_penalty_terms if term in outlet_keywords)
                if generic_matches > 0 and outlet_focus != specialization_type:
                    # REDUCED penalty for outlets that only have generic terms
                    score = score * (0.6 ** generic_matches)  # Increased from 0.3 - less strict
                
                # Additional penalty for marketing/retail/payments outlets in technical content - REDUCED PENALTY
                if specialization_type in ['cybersecurity', 'ai_ml', 'fintech', 'healthcare'] and outlet_focus in ['marketing', 'retail']:
                    score = score * 0.2  # Increased from 0.005 - less aggressive exclusion
                
                # Additional penalty for broad/generic outlets - REDUCED PENALTY
                broad_outlet_indicators = ['techradar', 'techcrunch', 'the next web', 'zdnet', 'infoworld', 'martech series', 'payments dive', 'retail touchpoints']
                if any(indicator in outlet.get('Outlet Name', '') for indicator in broad_outlet_indicators) and specialization_type in ['cybersecurity', 'ai_ml']:
                    score = score * 0.4  # Increased from 0.2 - less aggressive penalty
                
                # CLIENT'S REQUIREMENT: Downrank general/business outlets - REDUCED PENALTY
                general_business_outlets = ['time', 'inc', 'fortune', 'forbes', 'business insider', 'cnbc', 'bloomberg']
                if any(indicator in outlet.get('Outlet Name', '').lower() for indicator in general_business_outlets):
                    if specialization_type in ['cybersecurity', 'ai_ml', 'education', 'healthcare']:
                        score = score * 0.001  # 99.9% reduction for general business outlets in technical content
                    else:
                        score = score * 0.1  # 90% reduction for general business outlets
                
                # CLIENT FEEDBACK: Additional aggressive penalties for off-topic outlets
                if specialization_type in ['cybersecurity', 'ai_ml']:
                    outlet_name = outlet.get('Outlet Name', '').lower()
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    all_outlet_text = f"{outlet_name} {outlet_keywords} {outlet_audience}"
                    
                    # Check for specific off-topic outlets
                    off_topic_outlets = ['fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'food processing', 'retail touchpoints']
                    if any(off_topic in outlet_name for off_topic in off_topic_outlets):
                        score = score * 0.001  # 99.9% reduction
                    
                    # Check for off-topic keyword categories
                    fintech_keywords = ['fintech', 'payments', 'banking', 'financial', 'digital banking', 'mobile payments']
                    marketing_keywords = ['marketing', 'advertising', 'media', 'brand', 'campaign', 'creative']
                    business_keywords = ['business', 'corporate', 'enterprise', 'leadership', 'management']
                    retail_keywords = ['retail', 'commerce', 'ecommerce', 'shopping', 'consumer']
                    
                    total_off_topic_matches = 0
                    for keyword_list in [fintech_keywords, marketing_keywords, business_keywords, retail_keywords]:
                        total_off_topic_matches += sum(1 for kw in keyword_list if kw in all_outlet_text)
                    
                    if total_off_topic_matches >= 3:
                        normalized_score = normalized_score * 0.001  # 99.9% reduction
                    elif total_off_topic_matches >= 2:
                        normalized_score = normalized_score * 0.01  # 99% reduction
                    elif total_off_topic_matches >= 1:
                        normalized_score = normalized_score * 0.1  # 90% reduction
            
            # Boost score for specialized outlets (existing logic)
            if outlet.get('Outlet Name') in self.learned_data['outlet_specializations']:
                specialization = self.learned_data['outlet_specializations'][outlet.get('Outlet Name')]
                if industry.lower() in ' '.join(specialization['topics']).lower():
                    score = min(1.0, score * 2.5)  # Increased from 2.0 - much higher bonus
            
            return score
            
        except Exception as e:
            print(f"Error calculating industry match: {str(e)}")
            return 0.0

    def _calculate_keyword_relevance(self, outlet: Dict, query: str) -> float:
        """Calculate keyword relevance with semantic analysis."""
        try:
            # Process query and outlet keywords
            if self.nlp is None:
                # Fallback when spaCy is not available
                query_lower = query.lower()
                outlet_keywords = outlet.get('Keywords', '').lower()
                
                # Simple keyword overlap
                query_words = set(query_lower.split())
                outlet_words = set(outlet_keywords.split(','))
                overlap_score = len(query_words & outlet_words) / len(query_words | outlet_words) if outlet_words else 0
                
                return overlap_score
            
            query_doc = self.nlp(query.lower())
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_doc = self.nlp(outlet_keywords)
            
            # Detect specialized content across all categories
            content_specialization = self._detect_content_specialization(query, "")
            
            # Determine outlet focus
            outlet_focus = self._determine_outlet_focus(outlet)
            
            # Calculate semantic similarity
            semantic_score = query_doc.similarity(outlet_doc)
            
            # Calculate keyword overlap with improved matching
            query_words = set(query.lower().split())
            outlet_words = set(outlet_keywords.split(','))
            overlap_score = len(query_words & outlet_words) / len(query_words | outlet_words) if outlet_words else 0
            
            # Apply specialization penalties and bonuses with CLIENT'S STRICT REQUIREMENTS
            if content_specialization:
                specialization_type = content_specialization['type']
                
                # Define focus compatibility matrix for keyword relevance - CLIENT'S STRICT REQUIREMENTS
                focus_compatibility = {
                    'cybersecurity': ['cybersecurity'],  # Only cybersecurity specialists
                    'ai_ml': ['ai_ml'],                  # Only AI specialists
                    'fintech': ['fintech', 'finance'],    # Fintech or finance specialists only
                    'startup': ['startup', 'business', 'investment'],
                    'investment': ['investment', 'business', 'finance'],
                    'marketing': ['marketing'],           # Only marketing specialists
                    'healthcare': ['healthcare'],         # Only healthcare specialists
                    'education': ['education'],           # Only education specialists
                    'sustainability': ['sustainability'], # Only sustainability specialists
                    'real_estate': ['real_estate'],       # Only real estate specialists
                    'leadership': ['leadership', 'management', 'executive', 'ceo', 'c-suite', 'corporate strategy', 'organizational development', 'change management', 'talent management', 'workplace culture', 'diversity inclusion', 'remote work'],
                }
                
                compatible_focuses = focus_compatibility.get(specialization_type, [])
                
                if outlet_focus in compatible_focuses:
                    if outlet_focus == specialization_type:
                        # Perfect match - specialist outlet for specialist content
                        semantic_score = min(1.0, semantic_score * 1.8)  # Reduced from 2.0 - more reasonable
                        overlap_score = min(1.0, overlap_score * 1.8)
                    elif outlet_focus == 'tech_general' and specialization_type in ['cybersecurity', 'ai_ml']:
                        # Tech outlet for tech-specialized content - moderate score
                        semantic_score = semantic_score * 0.5  # Increased from 0.3 - less strict
                        overlap_score = overlap_score * 0.5
                    elif outlet_focus == 'business' and specialization_type in ['startup', 'investment', 'leadership']:
                        # Business outlet for business-specialized content
                        semantic_score = semantic_score * 0.6  # Increased from 0.4 - less strict
                        overlap_score = overlap_score * 0.6
                    elif outlet_focus == 'finance' and specialization_type == 'fintech':
                        # Finance outlet for fintech content
                        semantic_score = semantic_score * 0.6  # Increased from 0.4 - less strict
                        overlap_score = overlap_score * 0.6
                    else:
                        # Compatible but not perfect match - moderate score
                        semantic_score = semantic_score * 0.4  # Increased from 0.2 - less strict
                        overlap_score = overlap_score * 0.4
                else:
                    # Incompatible focus - REDUCED PENALTY
                    semantic_score = semantic_score * 0.1  # Increased from 0.01 - less aggressive exclusion
                    overlap_score = overlap_score * 0.1
                
                # CLIENT'S REQUIREMENT: Downrank/exclude irrelevant verticals - REDUCED PENALTY
                irrelevant_verticals = {
                    'cybersecurity': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'ai_ml': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'fintech': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'startup': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'investment': ['construction', 'healthcare', 'supply chain', 'retail', 'real estate'],
                    'marketing': ['construction', 'healthcare', 'banking', 'supply chain', 'real estate'],
                    'healthcare': ['construction', 'banking', 'supply chain', 'retail', 'real estate'],
                    'education': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate'],
                    'sustainability': ['construction', 'healthcare', 'banking', 'supply chain', 'retail'],
                    'real_estate': ['construction', 'healthcare', 'banking', 'supply chain', 'retail'],
                    'leadership': ['construction', 'healthcare', 'banking', 'supply chain', 'retail', 'real estate']
                }
                
                irrelevant_terms = irrelevant_verticals.get(specialization_type, [])
                if any(term in outlet_keywords for term in irrelevant_terms):
                    semantic_score = semantic_score * 0.3  # Increased from 0.1 - less aggressive penalty
                    overlap_score = overlap_score * 0.3
                
                # Additional penalty for generic terms in specialized content - REDUCED PENALTY
                generic_terms = {
                    'cybersecurity': ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'ai_ml': ['technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'fintech': ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce'],
                    'startup': ['business', 'technology', 'digital', 'marketing', 'advertising'],
                    'investment': ['business', 'finance', 'technology', 'marketing', 'advertising'],
                    'marketing': ['business', 'technology', 'digital'],
                    'healthcare': ['technology', 'digital', 'software', 'marketing', 'advertising'],
                    'education': ['technology', 'digital', 'software', 'marketing', 'advertising'],
                    'sustainability': ['technology', 'digital', 'business', 'marketing', 'advertising'],
                    'real_estate': ['technology', 'digital', 'business', 'marketing', 'advertising'],
                    'leadership': ['business', 'technology', 'marketing', 'advertising']
                }
                
                generic_penalty_terms = generic_terms.get(specialization_type, [])
                generic_matches = sum(1 for term in generic_penalty_terms if term in outlet_keywords)
                if generic_matches > 0 and outlet_focus != specialization_type:
                    # REDUCED penalty for outlets that only have generic terms
                    semantic_score = semantic_score * (0.5 ** generic_matches)  # Increased from 0.2 - less strict
                    overlap_score = overlap_score * (0.5 ** generic_matches)
                
                # Additional penalty for marketing/retail/payments outlets in technical content - REDUCED PENALTY
                if specialization_type in ['cybersecurity', 'ai_ml', 'fintech', 'healthcare'] and outlet_focus in ['marketing', 'retail']:
                    semantic_score = semantic_score * 0.3  # Increased from 0.002 - less aggressive exclusion
                    overlap_score = overlap_score * 0.3
                
                # Additional penalty for broad/generic outlets - REDUCED PENALTY
                outlet_name = outlet.get('Outlet Name', '').lower()
                broad_outlet_indicators = ['techradar', 'techcrunch', 'the next web', 'zdnet', 'infoworld', 'martech series', 'payments dive', 'retail touchpoints']
                if any(indicator in outlet_name for indicator in broad_outlet_indicators) and specialization_type in ['cybersecurity', 'ai_ml']:
                    semantic_score = semantic_score * 0.5  # Increased from 0.15 - less aggressive penalty
                    overlap_score = overlap_score * 0.5
                
                # CLIENT'S REQUIREMENT: Downrank general/business outlets - REDUCED PENALTY
                general_business_outlets = ['time', 'inc', 'fortune', 'forbes', 'business insider', 'cnbc', 'bloomberg']
                if any(indicator in outlet_name for indicator in general_business_outlets):
                    if specialization_type in ['cybersecurity', 'ai_ml', 'education', 'healthcare']:
                        semantic_score = semantic_score * 0.001  # 99.9% reduction for general business outlets in technical content
                        overlap_score = overlap_score * 0.001
                    else:
                        semantic_score = semantic_score * 0.1  # 90% reduction for general business outlets
                        overlap_score = overlap_score * 0.1
                
                # CLIENT FEEDBACK: Additional aggressive penalties for off-topic outlets
                if specialization_type in ['cybersecurity', 'ai_ml']:
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    all_outlet_text = f"{outlet_name} {outlet_keywords} {outlet_audience}"
                    
                    # Check for specific off-topic outlets
                    off_topic_outlets = ['fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'food processing', 'retail touchpoints']
                    if any(off_topic in outlet_name for off_topic in off_topic_outlets):
                        semantic_score = semantic_score * 0.001  # 99.9% reduction
                        overlap_score = overlap_score * 0.001
                    
                    # Check for off-topic keyword categories
                    fintech_keywords = ['fintech', 'payments', 'banking', 'financial', 'digital banking', 'mobile payments', 'payment processing']
                    marketing_keywords = ['marketing', 'advertising', 'media', 'brand', 'campaign', 'creative', 'digital marketing', 'social media marketing']
                    business_keywords = ['business', 'corporate', 'enterprise', 'leadership', 'management', 'strategy', 'business trends']
                    retail_keywords = ['retail', 'commerce', 'ecommerce', 'shopping', 'consumer', 'customer']
                    
                    # Apply penalties for off-topic keyword matches
                    fintech_matches = sum(1 for kw in fintech_keywords if kw in all_outlet_text)
                    marketing_matches = sum(1 for kw in marketing_keywords if kw in all_outlet_text)
                    business_matches = sum(1 for kw in business_keywords if kw in all_outlet_text)
                    retail_matches = sum(1 for kw in retail_keywords if kw in all_outlet_text)
                    
                    # Apply severe penalties for multiple off-topic keyword matches
                    total_off_topic_matches = fintech_matches + marketing_matches + business_matches + retail_matches
                    if total_off_topic_matches >= 3:
                        normalized_score = normalized_score * 0.001  # 99.9% reduction
                        
                    elif total_off_topic_matches >= 2:
                        normalized_score = normalized_score * 0.01  # 99% reduction
                        
                    elif total_off_topic_matches >= 1:
                        normalized_score = normalized_score * 0.1  # 90% reduction
                        
            
            # Combine scores with adjusted weights
            final_score = (semantic_score * 0.6) + (overlap_score * 0.4)  # Slightly favor semantic similarity
            
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

    def _calculate_news_match(self, outlet: Dict, query: str) -> float:
        """Calculate news match score based on recent articles."""
        try:
            outlet_name = outlet.get('Outlet Name')
            if not outlet_name or outlet_name not in self.learned_data['recent_articles']:
                return 0.0
                
            recent_articles = self.learned_data['recent_articles'][outlet_name]
            if not recent_articles:
                return 0.0
            
            if self.nlp is None:
                # Fallback when spaCy is not available
                query_lower = query.lower()
                max_similarity = 0.0
                
                for article in recent_articles:
                    article_text = f"{article.get('title', '')} {article.get('description', '')}".lower()
                    # Simple word overlap
                    query_words = set(query_lower.split())
                    article_words = set(article_text.split())
                    overlap = len(query_words & article_words) / len(query_words | article_words) if article_words else 0
                    max_similarity = max(max_similarity, overlap)
                
                return max_similarity
                
            # Calculate similarity with each article
            query_doc = self.nlp(query.lower())
            max_similarity = 0.0
            
            for article in recent_articles:
                article_text = f"{article.get('title', '')} {article.get('description', '')}"
                article_doc = self.nlp(article_text.lower())
                similarity = query_doc.similarity(article_doc)
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            print(f"Error calculating news match: {str(e)}")
            return 0.0

    def calculate_comprehensive_score_with_details(self, outlet: Dict, query: str, industry: str) -> Dict:
        """Calculate comprehensive match score with detailed field breakdown for debugging."""
        try:
            # Component scores
            industry_match = self._calculate_industry_match(outlet, industry)
            keyword_relevance = self._calculate_keyword_relevance(outlet, query)
            news_match = self._calculate_news_match(outlet, query)
            audience_match = self._calculate_audience_match(outlet, query, industry)
            content_compatibility = self._calculate_content_compatibility(outlet, query)
            requirements_match = self._calculate_requirements_match(outlet, query)
            outlet_expertise = self._calculate_outlet_expertise_match(outlet, query, industry)
            prestige_score = self._calculate_prestige_score(outlet)
            
            # Store individual scores for debugging
            field_scores = {
                'Industry Match': industry_match,
                'Keywords': keyword_relevance,
                'News Match': news_match,
                'Audience': audience_match,
                'Content Type': content_compatibility,
                'Requirements': requirements_match,
                'Outlet Expertise': outlet_expertise,
                'Prestige': prestige_score
            }
            
            # Weighted sum
            total_score = (
                industry_match * self.field_weights['Industry Match'] +
                keyword_relevance * self.field_weights['Keywords'] +
                news_match * self.field_weights['News Match'] +
                audience_match * self.field_weights['Audience'] +
                content_compatibility * self.field_weights['Content Type'] +
                requirements_match * self.field_weights['Requirements'] +
                outlet_expertise * self.field_weights['Outlet Expertise'] +
                prestige_score * self.field_weights['Prestige']
            )
            
            # Normalize score
            total_weight = sum(self.field_weights.values())
            normalized_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # Apply stricter exclusion logic for off-topic verticals
            content_specialization = self._detect_content_specialization(query, industry)
            outlet_focus = self._determine_outlet_focus(outlet)
            
            # CLIENT FEEDBACK: Stronger exclusion for off-topic verticals
            if content_specialization:
                specialization_type = content_specialization['type']
                
                # Define incompatible outlet focuses for each specialization - EXPANDED FOR CLIENT FEEDBACK
                incompatible_focuses = {
                    'education': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'cybersecurity': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'ai_ml': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'fintech': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'healthcare': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'sustainability': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'real_estate': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'startup': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'investment': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'marketing': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'leadership': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics']
                }
                
                # Check for incompatible outlet focus
                incompatible_list = incompatible_focuses.get(specialization_type, [])
                if outlet_focus in incompatible_list:
                    # CLIENT FEEDBACK: Score close to zero for incompatible verticals
                    normalized_score = normalized_score * 0.01  # Almost zero score
                
                # CLIENT FEEDBACK: Expanded specific outlet blacklists for security/AI content
                outlet_name = outlet.get('Outlet Name', '').lower()
                off_topic_outlets = {
                    'education': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'cybersecurity': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire', 'fintech', 'payments', 'marketing', 'advertising', 'business'],
                    'ai_ml': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire', 'fintech', 'payments', 'marketing', 'advertising', 'business'],
                    'fintech': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'healthcare': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'sustainability': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'real_estate': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'startup': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'investment': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'marketing': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'leadership': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in']
                }
                
                off_topic_list = off_topic_outlets.get(specialization_type, [])
                if any(off_topic in outlet_name for off_topic in off_topic_list):
                    normalized_score = normalized_score * 0.01  # Almost zero score
                
                # CLIENT FEEDBACK: Additional penalty for broad business/marketing outlets in technical content
                broad_business_outlets = ['time', 'inc', 'fortune', 'forbes', 'business insider', 'cnbc', 'bloomberg', 'fast company', 'adweek', 'marketing week', 'campaign', 'ad age']
                if specialization_type in ['cybersecurity', 'ai_ml', 'healthcare', 'education'] and any(broad in outlet_name for broad in broad_business_outlets):
                    normalized_score = normalized_score * 0.001  # 99.9% reduction for broad business outlets in technical content
                
                # CLIENT FEEDBACK: Additional penalty for fintech/payments outlets in non-fintech content
                fintech_outlets = ['fintech magazine', 'payments dive', 'payments source', 'banking dive', 'financial times', 'american banker']
                if specialization_type not in ['fintech'] and any(fintech in outlet_name for fintech in fintech_outlets):
                    normalized_score = normalized_score * 0.001  # 99.9% reduction for fintech outlets in non-fintech content
                
                # CLIENT FEEDBACK: Additional keyword-based penalties for technical content
                if specialization_type in ['cybersecurity', 'ai_ml']:
                    # Check for off-topic keywords in outlet data
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    all_outlet_text = f"{outlet_name} {outlet_keywords} {outlet_audience}"
                    
                    # Define off-topic keyword categories
                    fintech_keywords = ['fintech', 'payments', 'banking', 'financial', 'digital banking', 'mobile payments']
                    marketing_keywords = ['marketing', 'advertising', 'media', 'brand', 'campaign', 'creative']
                    business_keywords = ['business', 'corporate', 'enterprise', 'leadership', 'management']
                    retail_keywords = ['retail', 'commerce', 'ecommerce', 'shopping', 'consumer']
                    
                    # Apply penalties for off-topic keyword matches
                    fintech_matches = sum(1 for kw in fintech_keywords if kw in all_outlet_text)
                    marketing_matches = sum(1 for kw in marketing_keywords if kw in all_outlet_text)
                    business_matches = sum(1 for kw in business_keywords if kw in all_outlet_text)
                    retail_matches = sum(1 for kw in retail_keywords if kw in all_outlet_text)
                    
                    # Apply severe penalties for multiple off-topic keyword matches
                    total_off_topic_matches = fintech_matches + marketing_matches + business_matches + retail_matches
                    if total_off_topic_matches >= 3:
                        normalized_score = normalized_score * 0.001  # 99.9% reduction
                    elif total_off_topic_matches >= 2:
                        normalized_score = normalized_score * 0.01  # 99% reduction
                    elif total_off_topic_matches >= 1:
                        normalized_score = normalized_score * 0.1  # 90% reduction
            
            # Boost the final score to increase the average match confidence
            boosted_score = normalized_score * 1.5
            
            return {
                'score': min(1.0, max(0.0, boosted_score)),
                'field_scores': field_scores,
                'content_specialization': content_specialization,
                'outlet_focus': outlet_focus
            }
        
        except Exception as e:
            print(f"Error calculating comprehensive score with details: {str(e)}")
            return {
                'score': 0.0,
                'field_scores': {},
                'content_specialization': None,
                'outlet_focus': None
            }

    def calculate_comprehensive_score(self, outlet: Dict, query: str, industry: str) -> float:
        """Calculate a comprehensive match score based on multiple weighted factors."""
        try:
            # Component scores
            industry_match = self._calculate_industry_match(outlet, industry)
            keyword_relevance = self._calculate_keyword_relevance(outlet, query)
            news_match = self._calculate_news_match(outlet, query)
            audience_match = self._calculate_audience_match(outlet, query, industry)
            content_compatibility = self._calculate_content_compatibility(outlet, query)
            requirements_match = self._calculate_requirements_match(outlet, query)
            outlet_expertise = self._calculate_outlet_expertise_match(outlet, query, industry)
            prestige_score = self._calculate_prestige_score(outlet)
            
            # Weighted sum
            total_score = (
                industry_match * self.field_weights['Industry Match'] +
                keyword_relevance * self.field_weights['Keywords'] +
                news_match * self.field_weights['News Match'] +
                audience_match * self.field_weights['Audience'] +
                content_compatibility * self.field_weights['Content Type'] +
                requirements_match * self.field_weights['Requirements'] +
                outlet_expertise * self.field_weights['Outlet Expertise'] +
                prestige_score * self.field_weights['Prestige']
            )
            
            # Normalize score
            total_weight = sum(self.field_weights.values())
            normalized_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # Apply stricter exclusion logic for off-topic verticals
            content_specialization = self._detect_content_specialization(query, industry)
            outlet_focus = self._determine_outlet_focus(outlet)
            
            # CLIENT FEEDBACK: Stronger exclusion for off-topic verticals
            if content_specialization:
                specialization_type = content_specialization['type']
                
                # Define incompatible outlet focuses for each specialization - EXPANDED FOR CLIENT FEEDBACK
                incompatible_focuses = {
                    'education': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'cybersecurity': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'ai_ml': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'fintech': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'healthcare': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'sustainability': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'real_estate': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics', 'fintech', 'payments', 'marketing', 'advertising', 'business_general'],
                    'startup': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'investment': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'marketing': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics'],
                    'leadership': ['food', 'retail', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics']
                }
                
                # Check for incompatible outlet focus
                incompatible_list = incompatible_focuses.get(specialization_type, [])
                if outlet_focus in incompatible_list:
                    # CLIENT FEEDBACK: Score close to zero for incompatible verticals
                    normalized_score = normalized_score * 0.01  # Almost zero score
                
                # CLIENT FEEDBACK: Expanded specific outlet blacklists for security/AI content
                outlet_name = outlet.get('Outlet Name', '').lower()
                off_topic_outlets = {
                    'education': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'cybersecurity': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire', 'fintech', 'payments', 'marketing', 'advertising', 'business'],
                    'ai_ml': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire', 'fintech', 'payments', 'marketing', 'advertising', 'business'],
                    'fintech': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'healthcare': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'sustainability': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'real_estate': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in', 'fintech magazine', 'payments dive', 'cmswire', 'adweek', 'fast company', 'business insider', 'cms wire'],
                    'startup': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'investment': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'marketing': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in'],
                    'leadership': ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in']
                }
                
                off_topic_list = off_topic_outlets.get(specialization_type, [])
                if any(off_topic in outlet_name for off_topic in off_topic_list):
                    normalized_score = normalized_score * 0.01  # Almost zero score
                
                # CLIENT FEEDBACK: Additional penalty for broad business/marketing outlets in technical content
                broad_business_outlets = ['time', 'inc', 'fortune', 'forbes', 'business insider', 'cnbc', 'bloomberg', 'fast company', 'adweek', 'marketing week', 'campaign', 'ad age']
                if specialization_type in ['cybersecurity', 'ai_ml', 'healthcare', 'education'] and any(broad in outlet_name for broad in broad_business_outlets):
                    normalized_score = normalized_score * 0.001  # 99.9% reduction for broad business outlets in technical content
                
                # CLIENT FEEDBACK: Additional penalty for fintech/payments outlets in non-fintech content
                fintech_outlets = ['fintech magazine', 'payments dive', 'payments source', 'banking dive', 'financial times', 'american banker']
                if specialization_type not in ['fintech'] and any(fintech in outlet_name for fintech in fintech_outlets):
                    normalized_score = normalized_score * 0.001  # 99.9% reduction for fintech outlets in non-fintech content
                
                # CLIENT FEEDBACK: Additional keyword-based penalties for technical content
                if specialization_type in ['cybersecurity', 'ai_ml']:
                    # Check for off-topic keywords in outlet data
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    all_outlet_text = f"{outlet_name} {outlet_keywords} {outlet_audience}"
                    
                    # Define off-topic keyword categories
                    fintech_keywords = ['fintech', 'payments', 'banking', 'financial', 'digital banking', 'mobile payments']
                    marketing_keywords = ['marketing', 'advertising', 'media', 'brand', 'campaign', 'creative']
                    business_keywords = ['business', 'corporate', 'enterprise', 'leadership', 'management']
                    retail_keywords = ['retail', 'commerce', 'ecommerce', 'shopping', 'consumer']
                    
                    # Apply penalties for off-topic keyword matches
                    fintech_matches = sum(1 for kw in fintech_keywords if kw in all_outlet_text)
                    marketing_matches = sum(1 for kw in marketing_keywords if kw in all_outlet_text)
                    business_matches = sum(1 for kw in business_keywords if kw in all_outlet_text)
                    retail_matches = sum(1 for kw in retail_keywords if kw in all_outlet_text)
                    
                    # Apply severe penalties for multiple off-topic keyword matches
                    total_off_topic_matches = fintech_matches + marketing_matches + business_matches + retail_matches
                    if total_off_topic_matches >= 3:
                        normalized_score = normalized_score * 0.001  # 99.9% reduction
                        
                    elif total_off_topic_matches >= 2:
                        normalized_score = normalized_score * 0.01  # 99% reduction
                        
                    elif total_off_topic_matches >= 1:
                        normalized_score = normalized_score * 0.1  # 90% reduction
                        
            
            # Boost the final score to increase the average match confidence
            boosted_score = normalized_score * 1.5
            
            return min(1.0, max(0.0, boosted_score))
        
        except Exception as e:
            print(f"Error calculating comprehensive score: {str(e)}")
            return 0.0

    def _generate_match_explanation(self, outlet: Dict, score: float, query: str, industry: str) -> str:
        """Generate clear and concise match explanation based on actual matching criteria."""
        try:
            reasons = []
            
            # Detect specialized content across all categories
            if self.nlp is not None:
                query_doc = self.nlp(query.lower())
            else:
                query_doc = None
            content_specialization = self._detect_content_specialization(query, industry)
            
            # Get outlet details
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_name = outlet.get('Outlet Name', '').lower()
            outlet_audience = outlet.get('Audience', '').lower()
            outlet_section = outlet.get('Section Name', '').lower()
            
            # Determine outlet focus
            outlet_focus = self._determine_outlet_focus(outlet)
            
            # CLIENT FEEDBACK: Add field-level debugging information
            # Calculate individual field scores for explanation
            industry_match = self._calculate_industry_match(outlet, industry)
            keyword_relevance = self._calculate_keyword_relevance(outlet, query)
            audience_match = self._calculate_audience_match(outlet, query, industry)
            outlet_expertise = self._calculate_outlet_expertise_match(outlet, query, industry)
            
            # Handle specialized content with specific explanations
            if content_specialization:
                specialization_type = content_specialization['type']
                matched_terms = content_specialization.get('matched_terms', [])
                
                # Provide specific editorial focus explanation based on client requirements
                if outlet_focus == specialization_type:
                    # Perfect match - explain why this specialist outlet is ideal
                    focus_explanations = {
                        'cybersecurity': 'Cybersecurity specialist with deep technical expertise',
                        'ai_ml': 'AI/ML specialist covering advanced algorithms and applications',
                        'fintech': 'Fintech specialist covering financial technology innovation',
                        'startup': 'Startup specialist covering entrepreneurship and venture capital',
                        'investment': 'Investment specialist covering venture capital and M&A',
                        'marketing': 'Marketing specialist covering advertising and brand strategy',
                        'healthcare': 'Healthcare specialist covering medical technology and patient care',
                        'education': 'Education specialist covering EdTech and learning innovation',
                        'sustainability': 'Sustainability specialist covering climate tech and ESG',
                        'real_estate': 'Real estate specialist covering property technology',
                        'leadership': 'Leadership specialist covering executive management'
                    }
                    reasons.append(focus_explanations.get(specialization_type, f"{specialization_type.title()} specialist"))
                    
                    # Add specific technical terms if available
                    if matched_terms:
                        # Filter out generic terms and focus on specific ones
                        specific_terms = [term for term in matched_terms if term not in ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce']]
                        if specific_terms:
                            reasons.append(f"Core expertise: {', '.join(specific_terms[:3])}")
                        else:
                            reasons.append(f"Primary topic: {', '.join(matched_terms[:2])}")
                
                elif outlet_focus == 'tech_general' and specialization_type in ['cybersecurity', 'ai_ml']:
                    # Tech outlet for technical content
                    reasons.append("Technology outlet with strong technical coverage")
                    if matched_terms:
                        reasons.append(f"Technical focus: {', '.join(matched_terms[:2])}")
                
                elif outlet_focus == 'business' and specialization_type in ['startup', 'investment', 'leadership']:
                    # Business outlet for business content
                    reasons.append("Business outlet with industry expertise")
                    if matched_terms:
                        reasons.append(f"Business focus: {', '.join(matched_terms[:2])}")
                
                elif outlet_focus == 'finance' and specialization_type == 'fintech':
                    # Finance outlet for fintech content
                    reasons.append("Financial services outlet with fintech coverage")
                    if matched_terms:
                        reasons.append(f"Fintech expertise: {', '.join(matched_terms[:2])}")
                
                else:
                    # Incompatible or weak match - explain why it's not ideal
                    if outlet_focus in ['marketing', 'retail'] and specialization_type in ['cybersecurity', 'ai_ml', 'fintech', 'healthcare']:
                        reasons.append("Limited technical coverage - primarily marketing focus")
                    elif outlet_focus == 'tech_general' and specialization_type not in ['cybersecurity', 'ai_ml']:
                        reasons.append("General tech outlet - may lack specialized expertise")
                    elif any(term in outlet_name for term in ['time', 'inc', 'fortune', 'forbes', 'business insider', 'cnbc', 'bloomberg']):
                        reasons.append("General business outlet - broad coverage")
                    else:
                        reasons.append(f"Industry match: {industry}")
                
                # Add specific keyword matches with detailed breakdown
                if outlet_keywords:
                    # Look for specific technical terms that match the content
                    specific_matches = []
                    generic_matches = []
                    
                    for term in matched_terms:
                        if term in outlet_keywords:
                            if term not in ['ai', 'technology', 'digital', 'software', 'tech', 'marketing', 'advertising', 'business', 'retail', 'payments', 'commerce']:
                                specific_matches.append(term)
                            else:
                                generic_matches.append(term)
                    
                    if specific_matches:
                        reasons.append(f"Specific expertise: {', '.join(specific_matches[:3])}")
                    
                    if generic_matches and not specific_matches:
                        reasons.append(f"General coverage: {', '.join(generic_matches[:2])}")
                
            else:
                # For general content, use standard explanations
                if industry_match > 0.8:
                    reasons.append(f"Strong industry fit: {industry}")
                elif industry_match > 0.5:
                    reasons.append(f"Industry coverage: {industry}")
                
                # Add topic matches for general content
                outlet_keywords_list = [kw.strip() for kw in outlet_keywords.split(',') if kw.strip()]
                matching_keywords = []
                for kw in outlet_keywords_list:
                    if self.nlp is not None:
                        kw_doc = self.nlp(kw)
                        similarity = query_doc.similarity(kw_doc)
                        if similarity > 0.8:  # High threshold for meaningful matches
                            matching_keywords.append(kw)
                    else:
                        # Fallback when spaCy is not available
                        if kw.lower() in query.lower():
                            matching_keywords.append(kw)
                
                if matching_keywords:
                    reasons.append(f"Topic expertise: {', '.join(matching_keywords[:2])}")
            
            # CLIENT FEEDBACK: Add field-level debugging information
            # Show which fields contributed to the match
            contributing_fields = []
            if industry_match > 0.3:
                contributing_fields.append(f"Industry: {industry_match:.1%}")
            if keyword_relevance > 0.3:
                contributing_fields.append(f"Keywords: {keyword_relevance:.1%}")
            if audience_match > 0.3:
                contributing_fields.append(f"Audience: {audience_match:.1%}")
            if outlet_expertise > 0.3:
                contributing_fields.append(f"Expertise: {outlet_expertise:.1%}")
            
            if contributing_fields:
                reasons.append(f"Match factors: {', '.join(contributing_fields)}")
            
            # Add audience match with specific details
            if outlet_audience:
                # Extract meaningful audience terms
                if self.nlp is not None and query_doc is not None:
                    audience_doc = self.nlp(outlet_audience)
                    audience_terms = set([token.lemma_ for token in audience_doc if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 3])
                    query_terms = set([token.lemma_ for token in query_doc if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 3])
                    audience_overlap = list(audience_terms & query_terms)
                    
                    # Only show audience match if it's specific and not generic
                    if audience_overlap and len(audience_overlap) >= 2:
                        # Filter out generic terms
                        specific_audience_terms = [term for term in audience_overlap if term not in ['professionals', 'executives', 'leaders', 'experts']]
                        if specific_audience_terms:
                            reasons.append(f"Target audience: {', '.join(specific_audience_terms[:2])}")
                        else:
                            # Show the original audience if no specific terms found
                            reasons.append(f"Audience: {outlet_audience}")
                else:
                    # Fallback when spaCy is not available
                    reasons.append(f"Audience: {outlet_audience}")
            
            # Add section match with specific details
            if outlet_section:
                # Check if section is relevant to the content
                section_relevance = False
                for content_type, indicators in self.content_types.items():
                    if any(ind in outlet_section for ind in indicators):
                        reasons.append(f"Content format: {content_type}")
                        section_relevance = True
                        break
                
                # If no standard content type match, show the actual section
                if not section_relevance and outlet_section:
                    reasons.append(f"Section: {outlet_section}")
            
            # Add recent coverage match only for strong matches
            news_score = self._calculate_news_match(outlet, query)
            if news_score > 0.8:
                reasons.append("Strong recent coverage")
            elif news_score > 0.6:
                reasons.append("Recent relevant coverage")
            
            # If no specific reasons found, provide a general match quality
            if not reasons:
                if score >= 0.8:
                    reasons.append("Excellent editorial fit")
                elif score >= 0.6:
                    reasons.append("Strong editorial fit")
                elif score >= 0.4:
                    reasons.append("Good potential fit")
                else:
                    reasons.append("Limited editorial fit")
            
            return " • ".join(reasons)
            
        except Exception as e:
            print(f"Error generating match explanation: {str(e)}")
            return "Match details unavailable"

    def _detect_content_specialization(self, query: str, industry: str) -> Optional[Dict]:
        """Detect specialized content across all categories."""
        try:
            query_lower = query.lower()
            industry_lower = industry.lower()
            
            # Define specialized content categories with their indicators
            specialization_categories = {
                'cybersecurity': {
                    'terms': ['cybersecurity', 'cyber', 'security', 'encryption', 'quantum', 'threat', 'malware', 'vulnerability', 'breach', 'infrastructure', 'network security', 'zero-day', 'phishing', 'ransomware', 'firewall', 'authentication', 'compliance', 'gdpr', 'sox', 'pci'],
                    'audience': ['cybersecurity experts', 'tech professionals']
                },
                'ai_ml': {
                    'terms': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'ai', 'ml', 'algorithm', 'data science', 'predictive analytics', 'natural language processing', 'nlp', 'computer vision', 'robotics', 'automation'],
                    'audience': ['tech professionals', 'business executives']
                },
                'fintech': {
                    'terms': ['fintech', 'financial technology', 'digital banking', 'payments', 'blockchain', 'cryptocurrency', 'crypto', 'digital wallet', 'mobile banking', 'regtech', 'insurtech', 'wealthtech', 'lending', 'investment tech'],
                    'audience': ['finance & fintech leaders', 'tech professionals', 'business executives']
                },
                'startup': {
                    'terms': ['startup', 'entrepreneurship', 'venture capital', 'vc', 'funding', 'seed round', 'series a', 'series b', 'accelerator', 'incubator', 'pitch deck', 'business model', 'product-market fit', 'scaling', 'growth hacking'],
                    'audience': ['startup founders & entrepreneurs', 'investors & analysts']
                },
                'investment': {
                    'terms': ['investment', 'venture capital', 'private equity', 'mergers acquisitions', 'm&a', 'ipo', 'public offering', 'market analysis', 'portfolio management', 'asset management', 'hedge fund', 'angel investor'],
                    'audience': ['investors & analysts', 'business executives']
                },
                'marketing': {
                    'terms': ['marketing', 'advertising', 'brand', 'content marketing', 'social media marketing', 'digital marketing', 'seo', 'sem', 'email marketing', 'influencer marketing', 'growth marketing', 'customer acquisition', 'conversion optimization'],
                    'audience': ['marketing & pr professionals', 'business executives']
                },
                'healthcare': {
                    'terms': ['healthcare', 'medical', 'health tech', 'telemedicine', 'digital health', 'patient care', 'medical devices', 'pharmaceuticals', 'biotech', 'clinical trials', 'healthcare it', 'medical ai', 'precision medicine'],
                    'audience': ['healthcare & health tech leaders', 'tech professionals']
                },
                'education': {
                    'terms': ['education', 'edtech', 'learning', 'online education', 'distance learning', 'curriculum', 'pedagogy', 'student engagement', 'academic', 'higher education', 'k-12', 'lifelong learning', 'skills development', 'stem', 'special education', 'special ed', 'robots', 'robotics', 'embodied ai', 'interactive learning', 'personalized learning', 'student support', 'diverse students', 'engagement', 'pilot programs'],
                    'audience': ['education & policy leaders', 'general public', 'educators', 'school administrators']
                },
                'sustainability': {
                    'terms': ['sustainability', 'climate', 'environmental', 'green tech', 'renewable energy', 'carbon footprint', 'esg', 'circular economy', 'clean energy', 'climate tech', 'environmental impact', 'green building'],
                    'audience': ['sustainability & climate leaders', 'business executives']
                },
                'real_estate': {
                    'terms': ['real estate', 'property', 'commercial real estate', 'residential', 'proptech', 'real estate technology', 'property management', 'real estate investment', 'construction tech', 'smart buildings', 'facility management'],
                    'audience': ['real estate & built environment', 'business executives']
                },
                'leadership': ['leadership', 'management', 'executive', 'ceo', 'c-suite', 'corporate strategy', 'organizational development', 'change management', 'talent management', 'workplace culture', 'diversity inclusion', 'remote work'],
            }
            
            # Check for specialized content
            for category, config in specialization_categories.items():
                # Check if query contains specialized terms
                term_matches = [term for term in config['terms'] if term in query_lower]
                if term_matches:
                    return {
                        'type': category,
                        'terms': config['terms'],
                        'matched_terms': term_matches,
                        'audience': config['audience']
                    }
                
                # Check if industry contains specialized terms
                industry_term_matches = [term for term in config['terms'] if term in industry_lower]
                if industry_term_matches:
                    return {
                        'type': category,
                        'terms': config['terms'],
                        'matched_terms': industry_term_matches,
                        'audience': config['audience']
                    }
            
            return None
            
        except Exception as e:
            print(f"Error detecting content specialization: {str(e)}")
            return None

    def _determine_outlet_focus(self, outlet: Dict) -> str:
        """Determine the primary editorial focus of an outlet."""
        try:
            outlet_name = outlet.get('Outlet Name', '').lower()
            outlet_keywords = outlet.get('Keywords', '').lower()
            outlet_audience = outlet.get('Audience', '').lower()
            outlet_section = outlet.get('Section Name', '').lower()
            
            # Combine all text for analysis
            all_text = f"{outlet_name} {outlet_keywords} {outlet_audience} {outlet_section}"
            
            # Define focus categories with their indicators - expanded for all audience categories
            focus_indicators = {
                'cybersecurity': ['cybersecurity', 'cyber', 'security', 'threat', 'malware', 'vulnerability', 'breach', 'infrastructure', 'network security', 'zero-day', 'phishing', 'ransomware', 'firewall', 'authentication', 'compliance', 'gdpr', 'sox', 'pci', 'dark reading', 'securityweek', 'cso online'],
                'tech_general': ['technology', 'tech', 'software', 'digital', 'ai', 'machine learning', 'cloud', 'saas', 'startup', 'innovation', 'techcrunch', 'the next web', 'infoworld', 'zdnet'],
                'ai_ml': ['artificial intelligence', 'machine learning', 'deep learning', 'neural networks', 'ai', 'ml', 'algorithm', 'data science', 'predictive analytics', 'natural language processing', 'nlp', 'computer vision', 'robotics', 'automation'],
                'business': ['business', 'enterprise', 'corporate', 'leadership', 'strategy', 'management', 'fortune', 'wall street journal', 'bloomberg', 'harvard business review'],
                'startup': ['startup', 'entrepreneurship', 'venture capital', 'vc', 'funding', 'seed round', 'series a', 'series b', 'accelerator', 'incubator', 'pitch deck', 'business model', 'product-market fit', 'scaling', 'growth hacking'],
                'investment': ['investment', 'venture capital', 'private equity', 'mergers acquisitions', 'm&a', 'ipo', 'public offering', 'market analysis', 'portfolio management', 'asset management', 'hedge fund', 'angel investor'],
                'marketing': ['marketing', 'advertising', 'media', 'content strategy', 'digital marketing', 'brand', 'social media marketing', 'seo', 'sem', 'email marketing', 'influencer marketing', 'growth marketing', 'customer acquisition', 'conversion optimization', 'marketingprofs', 'adweek', 'content marketing institute'],
                'finance': ['finance', 'financial', 'banking', 'payments', 'fintech', 'financial technology', 'digital banking', 'blockchain', 'cryptocurrency', 'crypto', 'digital wallet', 'mobile banking', 'regtech', 'insurtech', 'wealthtech', 'lending', 'investment tech', 'payments dive'],
                'fintech': ['fintech', 'financial technology', 'digital banking', 'payments', 'blockchain', 'cryptocurrency', 'crypto', 'digital wallet', 'mobile banking', 'regtech', 'insurtech', 'wealthtech', 'lending', 'investment tech'],
                'healthcare': ['healthcare', 'medical', 'health tech', 'patient care', 'telemedicine', 'digital health', 'medical devices', 'pharmaceuticals', 'biotech', 'clinical trials', 'healthcare it', 'medical ai', 'precision medicine', 'healthcare it news', 'modern healthcare', 'medcitynews'],
                'education': ['education', 'edtech', 'learning', 'online education', 'distance learning', 'curriculum', 'pedagogy', 'student engagement', 'academic', 'higher education', 'k-12', 'lifelong learning', 'skills development', 'stem', 'special education', 'special ed', 'robots', 'robotics', 'embodied ai', 'interactive learning', 'personalized learning', 'student support', 'diverse students', 'engagement', 'pilot programs'],
                'sustainability': ['sustainability', 'climate', 'environmental', 'green tech', 'renewable energy', 'carbon footprint', 'esg', 'circular economy', 'clean energy', 'climate tech', 'environmental impact', 'green building', 'greenbiz', 'environmental leader'],
                'real_estate': ['real estate', 'property', 'commercial real estate', 'residential', 'proptech', 'real estate technology', 'property management', 'real estate investment', 'construction tech', 'smart buildings', 'facility management'],
                'leadership': ['leadership', 'management', 'executive', 'ceo', 'c-suite', 'corporate strategy', 'organizational development', 'change management', 'talent management', 'workplace culture', 'diversity inclusion', 'remote work'],
                'retail': ['retail', 'commerce', 'ecommerce', 'shopping', 'retail touchpoints', 'martech series'],
                # CLIENT FEEDBACK: Add specific off-topic verticals that should be excluded
                'food': ['food', 'food processing', 'food safety', 'agriculture', 'farming', 'food industry', 'food manufacturing', 'food packaging', 'food distribution', 'food service', 'restaurant', 'catering', 'food processing magazine'],
                'construction': ['construction', 'building', 'infrastructure', 'engineering', 'architecture', 'construction management', 'construction technology', 'building materials', 'construction industry', 'construction news'],
                'manufacturing': ['manufacturing', 'production', 'industrial', 'factory', 'manufacturing technology', 'supply chain', 'logistics', 'manufacturing industry', 'industrial automation'],
                'automotive': ['automotive', 'car', 'vehicle', 'automobile', 'auto industry', 'automotive technology', 'transportation', 'mobility', 'automotive manufacturing'],
                'energy': ['energy', 'power', 'utilities', 'renewable energy', 'oil', 'gas', 'electricity', 'energy industry', 'energy technology', 'power generation'],
                'transportation': ['transportation', 'logistics', 'shipping', 'freight', 'transport', 'logistics industry', 'supply chain', 'transportation technology'],
                'hospitality': ['hospitality', 'hotel', 'tourism', 'travel', 'lodging', 'hospitality industry', 'hotel management', 'tourism industry'],
                'agriculture': ['agriculture', 'farming', 'agtech', 'agricultural', 'crop', 'livestock', 'agricultural technology', 'precision agriculture'],
                # CLIENT FEEDBACK: Add new focus categories for better exclusion
                'payments': ['payments', 'payment processing', 'payment technology', 'digital payments', 'mobile payments', 'payment systems', 'payment security', 'payment dive', 'payments source'],
                'advertising': ['advertising', 'ad', 'media buying', 'creative', 'brand advertising', 'digital advertising', 'programmatic advertising', 'ad tech', 'adweek', 'ad age', 'campaign'],
                'business_general': ['business', 'business trends', 'business strategy', 'business news', 'business insights', 'business analysis', 'business insider', 'fast company', 'inc', 'fortune', 'forbes', 'cnbc', 'bloomberg']
            }
            
            # Calculate scores for each focus area
            focus_scores = {}
            for focus, indicators in focus_indicators.items():
                score = 0
                for indicator in indicators:
                    if indicator in all_text:
                        # Weight by position - outlet name matches are more important
                        if indicator in outlet_name:
                            score += 3
                        elif indicator in outlet_keywords:
                            score += 2
                        elif indicator in outlet_audience:
                            score += 1.5
                        elif indicator in outlet_section:
                            score += 1
                        else:
                            score += 0.5
                focus_scores[focus] = score
            
            # Return the focus with the highest score, or None if no clear focus
            if focus_scores:
                max_focus = max(focus_scores, key=focus_scores.get)
                if focus_scores[max_focus] >= 2:  # Require a minimum score
                    return max_focus
            
            return None
            
        except Exception as e:
            print(f"Error determining outlet focus: {str(e)}")
            return None

    def get_outlets(self) -> List[Dict]:
        """Fetch outlets with error handling."""
        try:
            print("Fetching outlets from database...")
            response = self.supabase.table("outlets").select("*").execute()
            if not response.data:
                print("No outlets found in database")
                return []
                
            print(f"Successfully fetched {len(response.data)} outlets from database")
            return response.data
        except Exception as e:
            print(f"Error fetching outlets: {str(e)}")
            import traceback
            traceback.print_exc()
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
                    if self.nlp is not None:
                        audience_doc = self.nlp(audience)
                        specificity_terms = [token.text for token in audience_doc if not token.is_stop]
                    else:
                        # Fallback when spaCy is not available
                        specificity_terms = audience.split()
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
                if self.nlp is not None:
                    audience_doc = self.nlp(audience)
                    key_terms = [token.text for token in audience_doc if not token.is_stop]
                else:
                    # Fallback when spaCy is not available
                    key_terms = audience.split()
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
            print(f"\n=== STARTING MATCH SEARCH ===")
            print(f"Query: {query}")
            print(f"Industry: {industry}")
            
            # Test database connection first
            self.test_database_connection()
            
            outlets = self.get_outlets()
            if not outlets:
                print("No outlets found in database")
                return []

            print(f"Processing {len(outlets)} outlets...")

            # Detect specialized content and adjust threshold accordingly
            content_specialization = self._detect_content_specialization(query, industry)
            print(f"Content specialization detected: {content_specialization}")
            
            # CLIENT FEEDBACK: Higher minimum threshold (20-30% confidence)
            # Use much higher thresholds to filter out weak matches
            dynamic_threshold = 0.20  # 20% minimum confidence
            if content_specialization:
                specialization_type = content_specialization['type']
                
                # Define higher thresholds for different specialization types
                specialization_thresholds = {
                    'cybersecurity': 0.25,  # Higher threshold for specialized content
                    'ai_ml': 0.25,          # Higher threshold for specialized content
                    'fintech': 0.25,        # Higher threshold for specialized content
                    'startup': 0.20,        # Moderate threshold
                    'investment': 0.20,     # Moderate threshold
                    'marketing': 0.20,      # Moderate threshold
                    'healthcare': 0.25,     # Higher threshold for specialized content
                    'education': 0.25,      # Higher threshold for specialized content
                    'sustainability': 0.20, # Moderate threshold
                    'real_estate': 0.20,    # Moderate threshold
                    'leadership': 0.20      # Moderate threshold
                }
                
                dynamic_threshold = specialization_thresholds.get(specialization_type, 0.20)
            else:
                dynamic_threshold = 0.20  # 20% minimum confidence for general content

            print(f"Using threshold: {dynamic_threshold} ({dynamic_threshold * 100:.0f}% confidence)")

            matches = []
            processed_count = 0
            
            # CLIENT FEEDBACK: Early filtering to exclude off-topic outlets before scoring
            filtered_outlets = self._pre_filter_outlets(outlets, query, industry, content_specialization)
            print(f"Pre-filtered from {len(outlets)} to {len(filtered_outlets)} outlets")
            
            for outlet in filtered_outlets:
                processed_count += 1
                if processed_count % 50 == 0:  # Progress indicator
                    print(f"Processed {processed_count}/{len(filtered_outlets)} outlets...")
                
                # Use comprehensive scoring with details for debugging
                score_details = self.calculate_comprehensive_score_with_details(outlet, query, industry)
                score = score_details['score']
                
                if score >= dynamic_threshold:
                    display_score = min(100, score * 100)
                    explanation = self._generate_match_explanation(outlet, score, query, industry)
                    
                    # CLIENT FEEDBACK: Add field-level debugging information
                    field_breakdown = self._generate_field_breakdown(score_details)
                    
                    matches.append({
                        "outlet": outlet,
                        "score": score,
                        "match_confidence": f"{round(display_score)}%",
                        "match_explanation": explanation,
                        "field_breakdown": field_breakdown,  # New field for debugging
                        "content_specialization": content_specialization,
                        "outlet_focus": score_details.get('outlet_focus')
                    })

            # CLIENT FEEDBACK: Better ranking - prioritize topical matches over general business/tech outlets
            matches = self._prioritize_topical_matches(matches, content_specialization)
            
            print(f"Found {len(matches)} matches above threshold {dynamic_threshold}")
            
            # If no matches found, run debug analysis
            if len(matches) == 0:
                print("No matches found - running debug analysis...")
                self.debug_pitch_analysis(query, industry)
            
            # Return top matches
            return matches[:limit]

        except Exception as e:
            print(f"Error finding matches: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def _pre_filter_outlets(self, outlets: List[Dict], query: str, industry: str, content_specialization: Optional[Dict]) -> List[Dict]:
        """Pre-filter outlets to exclude obvious off-topic ones before scoring."""
        try:
            if not content_specialization:
                return outlets  # No filtering for general content
            
            specialization_type = content_specialization['type']
            filtered_outlets = []
            
            # CLIENT FEEDBACK: Aggressive keyword-based filtering for security/AI content
            if specialization_type in ['cybersecurity', 'ai_ml']:
                # Define keywords that indicate off-topic outlets
                off_topic_keywords = [
                    'fintech', 'payments', 'banking', 'financial', 'retail', 'commerce', 'ecommerce',
                    'food', 'agriculture', 'farming', 'construction', 'manufacturing', 'automotive',
                    'energy', 'transportation', 'logistics', 'hospitality', 'tourism', 'travel',
                    'marketing', 'advertising', 'media', 'brand', 'campaign', 'creative',
                    'business', 'corporate', 'enterprise', 'leadership', 'management'
                ]
                
                # Define specific outlet name patterns to exclude
                excluded_patterns = [
                    'fintech magazine', 'payments dive', 'payments source', 'banking dive',
                    'food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in',
                    'cmswire', 'cms wire', 'adweek', 'fast company', 'business insider',
                    'marketing week', 'campaign', 'ad age', 'financial times', 'american banker',
                    'time', 'inc', 'fortune', 'forbes', 'cnbc', 'bloomberg'
                ]
                
                for outlet in outlets:
                    outlet_name = outlet.get('Outlet Name', '').lower()
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    
                    # Check for excluded patterns in outlet name
                    if any(pattern in outlet_name for pattern in excluded_patterns):
                        print(f"Pre-filtered: {outlet_name} (excluded pattern)")
                        continue
                    
                    # Check for off-topic keywords in outlet data
                    off_topic_matches = sum(1 for keyword in off_topic_keywords if keyword in outlet_name or keyword in outlet_keywords or keyword in outlet_audience)
                    
                    # If outlet has multiple off-topic indicators, exclude it
                    if off_topic_matches >= 2:
                        print(f"Pre-filtered: {outlet_name} ({off_topic_matches} off-topic indicators)")
                        continue
                    
                    # Check outlet focus
                    outlet_focus = self._determine_outlet_focus(outlet)
                    incompatible_focuses = ['fintech', 'payments', 'marketing', 'advertising', 'business_general', 'retail', 'food', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics']
                    
                    if outlet_focus in incompatible_focuses:
                        print(f"Pre-filtered: {outlet_name} (incompatible focus: {outlet_focus})")
                        continue
                    
                    filtered_outlets.append(outlet)
            
            # CLIENT FEEDBACK: Similar filtering for other specialized content
            elif specialization_type in ['education', 'healthcare']:
                # Define keywords that indicate off-topic outlets for education/healthcare
                off_topic_keywords = [
                    'fintech', 'payments', 'banking', 'financial', 'retail', 'commerce', 'ecommerce',
                    'food', 'agriculture', 'farming', 'construction', 'manufacturing', 'automotive',
                    'energy', 'transportation', 'logistics', 'hospitality', 'tourism', 'travel',
                    'marketing', 'advertising', 'media', 'brand', 'campaign', 'creative'
                ]
                
                # Define specific outlet name patterns to exclude
                excluded_patterns = [
                    'fintech magazine', 'payments dive', 'payments source', 'banking dive',
                    'food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in',
                    'cmswire', 'cms wire', 'adweek', 'fast company', 'business insider',
                    'marketing week', 'campaign', 'ad age', 'financial times', 'american banker'
                ]
                
                for outlet in outlets:
                    outlet_name = outlet.get('Outlet Name', '').lower()
                    outlet_keywords = outlet.get('Keywords', '').lower()
                    outlet_audience = outlet.get('Audience', '').lower()
                    
                    # Check for excluded patterns in outlet name
                    if any(pattern in outlet_name for pattern in excluded_patterns):
                        print(f"Pre-filtered: {outlet_name} (excluded pattern)")
                        continue
                    
                    # Check for off-topic keywords in outlet data
                    off_topic_matches = sum(1 for keyword in off_topic_keywords if keyword in outlet_name or keyword in outlet_keywords or keyword in outlet_audience)
                    
                    # If outlet has multiple off-topic indicators, exclude it
                    if off_topic_matches >= 2:
                        print(f"Pre-filtered: {outlet_name} ({off_topic_matches} off-topic indicators)")
                        continue
                    
                    # Check outlet focus
                    outlet_focus = self._determine_outlet_focus(outlet)
                    incompatible_focuses = ['fintech', 'payments', 'marketing', 'advertising', 'business_general', 'retail', 'food', 'construction', 'manufacturing', 'agriculture', 'hospitality', 'automotive', 'energy', 'transportation', 'logistics']
                    
                    if outlet_focus in incompatible_focuses:
                        print(f"Pre-filtered: {outlet_name} (incompatible focus: {outlet_focus})")
                        continue
                    
                    filtered_outlets.append(outlet)
            
            else:
                # For other specializations, use basic filtering
                for outlet in outlets:
                    outlet_name = outlet.get('Outlet Name', '').lower()
                    
                    # Basic pattern exclusion
                    excluded_patterns = ['food processing', 'retail touchpoints', 'pr daily', 'sc magazine', 'built in']
                    if any(pattern in outlet_name for pattern in excluded_patterns):
                        print(f"Pre-filtered: {outlet_name} (basic excluded pattern)")
                        continue
                    
                    filtered_outlets.append(outlet)
            
            return filtered_outlets
            
        except Exception as e:
            print(f"Error in pre-filtering: {str(e)}")
            return outlets  # Return all outlets if filtering fails

    def _generate_field_breakdown(self, score_details: Dict) -> str:
        """Generate field-level breakdown for debugging."""
        try:
            field_scores = score_details.get('field_scores', {})
            if not field_scores:
                return "No field scores available"
            
            # Filter out zero scores and sort by importance
            non_zero_fields = {k: v for k, v in field_scores.items() if v > 0.01}
            
            if not non_zero_fields:
                return "All field scores below threshold"
            
            # Sort by score value (descending)
            sorted_fields = sorted(non_zero_fields.items(), key=lambda x: x[1], reverse=True)
            
            # Format the breakdown
            breakdown_parts = []
            for field_name, score in sorted_fields[:3]:  # Show top 3 fields
                percentage = score * 100
                breakdown_parts.append(f"{field_name}: {percentage:.1f}%")
            
            return " | ".join(breakdown_parts)
            
        except Exception as e:
            print(f"Error generating field breakdown: {str(e)}")
            return "Field breakdown unavailable"

    def _prioritize_topical_matches(self, matches: List[Dict], content_specialization: Optional[Dict]) -> List[Dict]:
        """Sort matches by score descending only (no topical/general grouping)."""
        try:
            matches.sort(key=lambda x: -x['score'])
            return matches
        except Exception as e:
            print(f"Error prioritizing matches by score: {str(e)}")
            return matches

    def update_recent_articles(self, outlet_name: str, articles: List[Dict]):
        """Update recent articles for an outlet."""
        try:
            self.learned_data['recent_articles'][outlet_name] = articles
        except Exception as e:
            print(f"Error updating recent articles: {str(e)}")

    def test_database_connection(self):
        """Test database connection and outlet data."""
        try:
            print("Testing database connection...")
            outlets = self.get_outlets()
            print(f"Database connection successful. Found {len(outlets)} outlets.")
            
            if len(outlets) > 0:
                print("Sample outlet data:")
                sample_outlet = outlets[0]
                print(f"  Name: {sample_outlet.get('Outlet Name', 'N/A')}")
                print(f"  Keywords: {sample_outlet.get('Keywords', 'N/A')}")
                print(f"  Audience: {sample_outlet.get('Audience', 'N/A')}")
                print(f"  Prestige: {sample_outlet.get('Prestige', 'N/A')}")
            else:
                print("WARNING: No outlets found in database!")
                
        except Exception as e:
            print(f"Database connection failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def debug_pitch_analysis(self, query: str, industry: str):
        """Debug method to analyze why a pitch isn't returning results."""
        print(f"\n=== QUICK DEBUG ANALYSIS ===")
        print(f"Query: {query}")
        print(f"Industry: {industry}")
        
        # Check content specialization
        content_specialization = self._detect_content_specialization(query, industry)
        print(f"Content specialization: {content_specialization}")
        
        # Get outlets
        outlets = self.get_outlets()
        print(f"Total outlets: {len(outlets)}")
        
        if len(outlets) == 0:
            print("ERROR: No outlets found in database!")
            return
        
        # Check first 3 outlets only for speed
        print(f"\n--- Checking first 3 outlets ---")
        for i, outlet in enumerate(outlets[:3]):
            outlet_name = outlet.get('Outlet Name', 'Unknown')
            print(f"\nOutlet {i+1}: {outlet_name}")
            
            # Check basic data
            keywords = outlet.get('Keywords', '')
            audience = outlet.get('Audience', '')
            print(f"  Keywords: {keywords}")
            print(f"  Audience: {audience}")
            
            # Calculate score
            score = self.calculate_similarity_score(outlet, query, industry)
            print(f"  Score: {score:.3f}")
            
            # Check for basic matches
            query_words = query.lower().split()
            keyword_matches = []
            for word in query_words:
                if word in keywords.lower():
                    keyword_matches.append(word)
            print(f"  Keyword matches: {keyword_matches}")
        
        print("=== END DEBUG ANALYSIS ===\n")