from typing import List, Dict, Set, Tuple
from supabase import Client
import re
import warnings
warnings.filterwarnings('ignore')

class OutletMatcher:
    """WriteFor.co Matching Logic v4 - Exact specification implementation."""
    
    # Core configuration
    MIN_RESULTS = 8
    AI_PARTNER_NULL_STATE = "Unconfirmed"
    
    # Outlet families mapping
    OUTLET_FAMILIES = {
        'Healthcare': ['Modern Healthcare', 'Healthcare IT News', 'HIT Consultant', 'Healthcare Innovation', 'MedCity News', 'Healthcare Design', 'Fierce Healthcare', 'Becker\'s Hospital Review', 'HealthLeaders', 'Healthcare Dive', 'Health Data Management', 'Healthcare Finance News', 'STAT News', 'HIMSS Media'],
        'Cybersecurity': ['Dark Reading', 'SC Magazine', 'SecurityWeek', 'Security Boulevard', 'The Hacker News', 'BleepingComputer', 'Threatpost', 'CSO Online', 'Information Security Magazine', 'Help Net Security', 'Security Intelligence'],
        'Developer/IT': ['InfoWorld', 'InfoQ', 'SD Times', 'The New Stack', 'DevOps.com', 'ITPro', 'Cloud Computing News', 'Opensource.com', 'IEEE Software', 'i-Programmer', 'ACM Queue', 'DZone', 'TechTarget'],
        'Business/TierOne': ['The Wall Street Journal', 'Financial Times', 'Bloomberg', 'The Economist', 'Fortune', 'Harvard Business Review', 'Business Insider', 'TIME', 'The Atlantic', 'Fast Company', 'Inc.'],
        'Retail': ['Retail Dive', 'Retail TouchPoints', 'Retail Touch Points'],
        'Food/CPG': ['Food Processing', 'Food Dive'],
        'SupplyChain/Ops': ['Supply Chain Dive', 'Supply Chain Management Review', 'SCM Review'],
        'HR/People': ['HR Dive', 'SHRM'],
        'Finance/Banking': ['American Banker', 'Banking Dive', 'Payments Dive', 'FinTech Magazine', 'PYMNTS'],
        'Energy/Climate': ['Energy Central', 'Environment+Energy Leader', 'Factor This!', 'Trellis', 'CleanTechnica', 'TreeHugger', 'EcoWatch', 'Renewable Energy World'],
        'Education': ['The Chronicle of Higher Education', 'EdTech Magazine', 'EdSurge', 'Campus Technology', 'eSchool News', 'Education Week', 'Inside Higher Ed'],
        'RealEstate/BuiltEnv': ['Inman', 'Construction Dive', 'Engineering News-Record', 'ENR'],
        'Lifestyle/Wellness': ['MindBodyGreen', 'Wellness Mama', 'Healthline', 'Prevention', 'Women\'s Health', 'Men\'s Health', 'Shape', 'Fitness', 'Yoga Journal'],
        'GeneralTech/Consumer': ['TechCrunch', 'Wired', 'The Verge', 'Ars Technica', 'Engadget', 'Gizmodo', 'Mashable', 'VentureBeat', 'The Next Web', 'Recode']
    }
    
    # Trigger dictionary (abstract keywords → families)
    TRIGGER_DICTIONARY = {
        'Cybersecurity': ['cybersecurity', 'security', 'ciso', 'ransomware', 'phishing', 'zero trust', 'soc', 'siem', 'threat', 'incident'],
        'Developer/IT': ['developer', 'software', 'engineering', 'it', 'cloud', 'devops', 'kubernetes', 'api', 'sdk', 'infrastructure', 'low-code', 'no-code'],
        'Finance/Banking': ['bank', 'banking', 'finance', 'fintech', 'payments', 'credit', 'lending', 'treasury'],
        'Healthcare': ['healthcare', 'hospital', 'patient', 'provider', 'medical', 'clinical', 'clinician', 'ehr', 'hipaa', 'pharma', 'biotech'],
        'Energy/Climate': ['energy', 'utility', 'utilities', 'climate', 'sustainability', 'esg', 'emissions', 'carbon', 'renewable', 'ev', 'battery', 'decarbonization', 'dle'],
        'Education': ['education', 'edtech', 'higher ed', 'university', 'school', 'student', 'teacher', 'academic integrity', 'plagiarism'],
        'Retail': ['retail', 'ecommerce', 'e-commerce', 'shopping', 'store', 'cpg', 'consumer goods', 'omnichannel'],
        'Food/CPG': ['food', 'agriculture', 'farming', 'beverage', 'dairy', 'packaged goods'],
        'SupplyChain/Ops': ['supply chain', 'logistics', 'shipping', 'warehousing', 'freight', 'distribution', 'transportation', 'last-mile'],
        'HR/People': ['hr', 'human resources', 'workforce', 'hiring', 'recruiting', 'retention', 'benefits', 'compensation', 'dei', 'labor'],
        'RealEstate/BuiltEnv': ['real estate', 'property', 'proptech', 'building', 'construction', 'infrastructure', 'cre'],
        'Lifestyle/Wellness': ['wellness', 'mental health', 'mindfulness', 'work-life', 'sleep', 'nutrition']
    }

    def __init__(self, supabase_client: Client):
        """Initialize the outlet matcher with v4 configuration."""
        self.supabase = supabase_client
        print("🔄 OutletMatcher v4 initialized - Exact specification implementation")
        
        # Load matching configuration
        self.matching_config = self._load_matching_config()
        
        # Initialize NLP for keyword matching
        self.nlp = self._initialize_nlp()

    def _load_matching_config(self) -> Dict:
        """Load the matching configuration from JSON file."""
        try:
            import os
            import json
            config_path = os.path.join(os.path.dirname(__file__), '..', 'matching_config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✅ Loaded matching configuration")
            return config
        except Exception as e:
            print(f"⚠️ Failed to load matching config: {e}")
            return {}

    def _initialize_nlp(self):
        """Initialize NLP for keyword matching with graceful fallback."""
        try:
            import spacy  # type: ignore
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy loaded for keyword matching")
            return nlp
        except ImportError:
            print("⚠️ spaCy not installed - using fallback keyword matching")
            return None
        except Exception as e:
            print(f"⚠️ spaCy model not found: {e} - using fallback keyword matching")
            return None

    def _hard_audience_filter(self, outlets: List[Dict], selected_audience: str) -> List[Dict]:
        """Hard audience pre-filter using Column M with exact token matching."""
        filtered_outlets = []
        selected_audience_lower = selected_audience.lower()
        
        for outlet in outlets:
            outlet_audience_tags = outlet.get('Industry', '')  # Column M
            
            if not outlet_audience_tags:
                continue
            
            # Split Column M by ;, trim, lowercase
            audience_tokens = [tag.strip().lower() for tag in outlet_audience_tags.split(';')]
            
            # Keep outlets where selected audience matches a full token (exact, case-insensitive)
            if selected_audience_lower in audience_tokens:
                filtered_outlets.append(outlet)
        
        print(f"🔍 Hard filter: {len(outlets)} → {len(filtered_outlets)} outlets for '{selected_audience}'")
        return filtered_outlets

    def _get_outlet_families(self, outlet_name: str) -> List[str]:
        """Assign outlet to families based on outlet name."""
        outlet_name_lower = outlet_name.lower()
        families = []
        
        for family, outlets in self.OUTLET_FAMILIES.items():
            for family_outlet in outlets:
                if family_outlet.lower() in outlet_name_lower or outlet_name_lower in family_outlet.lower():
                    families.append(family)
                    break
        
        return families

    def _count_trigger_hits(self, abstract: str, family: str) -> int:
        """Count trigger hits for a specific family in the abstract."""
        if family not in self.TRIGGER_DICTIONARY:
            return 0
        
        abstract_lower = abstract.lower()
        triggers = self.TRIGGER_DICTIONARY[family]
        hits = 0
        
        for trigger in triggers:
            if trigger.lower() in abstract_lower:
                hits += 1
        
        return hits

    def _count_keyword_matches(self, abstract: str, outlet_keywords: str) -> int:
        """Count keyword matches between abstract and outlet keywords (Column G) using spaCy."""
        if not outlet_keywords or not abstract:
            return 0
        
        # Fallback to simple string matching if spaCy not available
        if not self.nlp:
            abstract_lower = abstract.lower()
            outlet_keywords_lower = outlet_keywords.lower()
            return 1 if any(keyword.strip().lower() in abstract_lower for keyword in outlet_keywords.split(',')) else 0
        
        try:
            # Use spaCy for better keyword matching
            abstract_doc = self.nlp(abstract.lower())
            outlet_keywords_list = [kw.strip().lower() for kw in outlet_keywords.split(',')]
            
            matches = 0
            for keyword in outlet_keywords_list:
                if not keyword:
                    continue
                
                # Check for exact matches and lemmatized matches
                keyword_doc = self.nlp(keyword)
                keyword_lemma = keyword_doc[0].lemma_ if len(keyword_doc) > 0 else keyword
                
                for token in abstract_doc:
                    if (token.text == keyword or 
                        token.lemma_ == keyword_lemma or 
                        keyword in token.text or 
                        token.text in keyword):
                        matches += 1
                        break
            
            return matches
        except Exception as e:
            print(f"⚠️ spaCy keyword matching failed: {e}, using fallback")
            # Fallback to simple matching
            abstract_lower = abstract.lower()
            outlet_keywords_lower = outlet_keywords.lower()
            return 1 if any(keyword.strip().lower() in abstract_lower for keyword in outlet_keywords.split(',')) else 0

    def _count_audience_matches(self, abstract: str, outlet_audience: str) -> int:
        """Count audience matches between abstract and outlet audience (Column F) using spaCy."""
        if not outlet_audience or not abstract:
            return 0
        
        # Fallback to simple string matching if spaCy not available
        if not self.nlp:
            abstract_lower = abstract.lower()
            outlet_audience_lower = outlet_audience.lower()
            return 1 if outlet_audience_lower in abstract_lower else 0
        
        try:
            # Use spaCy for better audience matching
            abstract_doc = self.nlp(abstract.lower())
            outlet_audience_lower = outlet_audience.lower()
            
            # Check for audience terms in abstract
            audience_terms = [term.strip().lower() for term in outlet_audience_lower.split(',')]
            
            matches = 0
            for audience_term in audience_terms:
                if not audience_term:
                    continue
                
                audience_doc = self.nlp(audience_term)
                audience_lemma = audience_doc[0].lemma_ if len(audience_doc) > 0 else audience_term
                
                for token in abstract_doc:
                    if (token.text == audience_term or 
                        token.lemma_ == audience_lemma or 
                        audience_term in token.text or 
                        token.text in audience_term):
                        matches += 1
                        break
            
            return matches
        except Exception as e:
            print(f"⚠️ spaCy audience matching failed: {e}, using fallback")
            # Fallback to simple matching
            abstract_lower = abstract.lower()
            outlet_audience_lower = outlet_audience.lower()
            return 1 if outlet_audience_lower in abstract_lower else 0

    def _compute_score(self, outlet: Dict, abstract: str, selected_audience: str) -> float:
        """Compute score using exact ranking logic with spaCy keyword matching."""
        outlet_name = outlet.get('Outlet Name', '')
        outlet_keywords = outlet.get('Keywords', '')  # Column G
        outlet_audience = outlet.get('Audience', '')  # Column F
        outlet_families = self._get_outlet_families(outlet_name)
        
        score = 0.0
        
        # +3.0 if outlet has a family that matches the selected audience
        if selected_audience in outlet_families:
            score += 3.0
        
        # +2.0 per primary-family trigger hit from the abstract (cap +6.0)
        primary_triggers = self._count_trigger_hits(abstract, selected_audience)
        score += min(primary_triggers * 2.0, 6.0)
        
        # +1.0 per secondary-family trigger (families present on outlet but not selected audience, cap +2.0)
        secondary_score = 0.0
        for family in outlet_families:
            if family != selected_audience:
                family_triggers = self._count_trigger_hits(abstract, family)
                secondary_score += family_triggers * 1.0
        score += min(secondary_score, 2.0)
        
        # +1.0 per keyword match between abstract and outlet keywords (Column G) using spaCy
        keyword_matches = self._count_keyword_matches(abstract, outlet_keywords)
        score += min(keyword_matches * 1.0, 3.0)  # Cap at +3.0
        
        # +0.5 per audience match between abstract and outlet audience (Column F) using spaCy
        audience_matches = self._count_audience_matches(abstract, outlet_audience)
        score += min(audience_matches * 0.5, 1.5)  # Cap at +1.5
        
        # -3.0 cross-family penalty: if outlet is in non-selected family and abstract has no triggers for that family
        for family in outlet_families:
            if family != selected_audience:
                family_triggers = self._count_trigger_hits(abstract, family)
                if family_triggers == 0:
                    score -= 3.0
        
        # Business Executives only: +1.8 if outlet is in tier-one list
        if selected_audience == "Business Executives":
            tier_one_outlets = ['The Wall Street Journal', 'Financial Times', 'Bloomberg', 'The Economist', 'Fortune', 'Harvard Business Review', 'Business Insider', 'TIME', 'The Atlantic', 'Fast Company', 'Inc.']
            if outlet_name in tier_one_outlets:
                score += 1.8
        
        return score

    def _normalize_scores(self, scored_results: List[Dict]) -> List[Dict]:
        """Normalize scores within the candidate set to 50-100, cap at 100."""
        if not scored_results:
            return scored_results
        
        scores = [result['score'] for result in scored_results]
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            normalized_results = [{'outlet': result['outlet'], 'score': 75.0} for result in scored_results]
        else:
            normalized_results = []
            for result in scored_results:
                # Normalize to 50-100 range instead of 0-100
                normalized_score = 50 + ((result['score'] - min_score) / (max_score - min_score)) * 50
                normalized_score = min(normalized_score, 100.0)  # Cap at 100
                normalized_results.append({
                    'outlet': result['outlet'],
                    'score': normalized_score
                })
        
        return normalized_results

    def find_matches_v4(self, abstract: str, industry: str, limit: int = 20, debug_mode: bool = False) -> List[Dict]:
        """V4 matching logic with exact specification implementation."""
        print(f"🎯 Starting v4 matching for '{industry}' audience")
        
        # Get all outlets
        all_outlets = self.get_outlets()
        if not all_outlets:
            print("❌ No outlets found")
            return []
        
        # Step 1: Hard audience pre-filter (Column M)
        filtered_outlets = self._hard_audience_filter(all_outlets, industry)
        
        # If zero remain, show empty state (do not fall back to keyword-only)
        if not filtered_outlets:
            print(f"❌ No outlets found for audience '{industry}' - showing empty state")
            return []
        
        # Step 2: Score all candidates
        scored_results = []
        for outlet in filtered_outlets:
            score = self._compute_score(outlet, abstract, industry)
            scored_results.append({
                'outlet': outlet,
                'score': score
            })
        
        # Step 3: Normalize scores to 0-100
        normalized_results = self._normalize_scores(scored_results)
        
        # Step 4: Sort by score desc; tie → outlet name asc (deterministic)
        normalized_results.sort(key=lambda x: (-x['score'], x['outlet'].get('Outlet Name', '')))
        
        # Step 5: Limit results
        final_results = normalized_results[:limit]
        
        print(f"✅ Returning {len(final_results)} results for '{industry}' audience")
        
        # Format results to match expected structure
        formatted_results = []
        for result in final_results:
            outlet = result['outlet']
            score = result['score']
            
            formatted_results.append({
                'outlet': outlet,
                'score': score / 100.0,  # Convert to 0-1 range for internal use
                'match_confidence': f'{score:.1f}%',  # Display as percentage with 1 decimal
                'match_explanation': f"Audience: {industry} | Score: {score:.1f}/100 | Outlet: {outlet.get('Outlet Name', 'Unknown')}"
            })
        
        return formatted_results

    def find_matches(self, abstract: str, industry: str, limit: int = 20, debug_mode: bool = False) -> List[Dict]:
        """Main matching method - now uses v4 logic."""
        return self.find_matches_v4(abstract, industry, limit, debug_mode)

    def get_outlets(self) -> List[Dict]:
        """Get all outlets from Supabase."""
        try:
            response = self.supabase.table('outlets').select('*').execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"⚠️ Failed to get outlets: {e}")
            return []