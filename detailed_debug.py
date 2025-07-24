import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
from models.matcher import OutletMatcher

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize matcher
matcher = OutletMatcher(supabase)

# Test data
test_abstract = "Our AI-powered learning platform helps students improve their math skills through personalized tutoring and adaptive assessments."
test_industry = "Education & Policy Leaders"

print("🔍 DETAILED DEBUG ANALYSIS")
print("=" * 60)
print(f"Abstract: {test_abstract}")
print(f"Industry: '{test_industry}'")
print()

# Get all outlets
outlets = matcher.get_outlets()
print(f"Total outlets found: {len(outlets)}")

# Test first 10 outlets with detailed analysis
print("\n📊 DETAILED ANALYSIS OF FIRST 10 OUTLETS:")
print("=" * 60)

for i, outlet in enumerate(outlets[:10], 1):
    name = outlet.get('Outlet Name', 'Unknown')
    audience = outlet.get('Audience', 'N/A')
    keywords = outlet.get('Keywords', 'N/A')
    section = outlet.get('Section Name', 'N/A')
    pitch_tips = outlet.get('Pitch Tips', 'N/A')
    prestige = outlet.get('Prestige', 'N/A')
    ai_partnered = outlet.get('AI Partnered', 'N/A')
    
    print(f"\n{i}. {name}")
    print(f"   Audience: {audience}")
    print(f"   Keywords: {keywords}")
    print(f"   Section: {section}")
    print(f"   Pitch Tips: {pitch_tips}")
    print(f"   Prestige: {prestige}")
    print(f"   AI Partnered: {ai_partnered}")
    
    # Test exact industry matching
    exact_score = matcher._calculate_exact_industry_match(outlet, test_industry)
    print(f"   🔍 Exact Industry Match: {exact_score:.3f}")
    
    # Test semantic similarity
    combined_text = f"{test_abstract} {test_industry}".lower()
    
    # Test each column individually
    if audience:
        audience_sim = matcher._calculate_semantic_similarity(test_industry, audience)
        print(f"   📈 Industry vs Audience: {audience_sim:.3f}")
    
    if keywords:
        keyword_sim = matcher._calculate_semantic_similarity(combined_text, keywords)
        print(f"   📈 Content vs Keywords: {keyword_sim:.3f}")
    
    if section:
        section_sim = matcher._calculate_semantic_similarity(combined_text, section)
        print(f"   📈 Content vs Section: {section_sim:.3f}")
    
    # Test abstract analysis
    abstract_analysis = matcher._analyze_abstract_meaning(test_abstract)
    print(f"   🧠 Abstract Analysis:")
    print(f"      Topics: {list(abstract_analysis['topics'])[:5]}")
    print(f"      Actions: {list(abstract_analysis['actions'])[:3]}")
    print(f"      Entities: {list(abstract_analysis['entities'])[:3]}")
    print(f"      Industry Keywords: {list(abstract_analysis['industry_keywords'])[:3]}")
    
    # Test semantic boost
    if keywords:
        semantic_boost = matcher._calculate_semantic_boost(abstract_analysis, keywords)
        print(f"   🚀 Semantic Boost: {semantic_boost:.3f}")
    
    print("-" * 60)

# Test full matching with debug output
print("\n🏆 FULL MATCHING RESULTS WITH DEBUG:")
print("=" * 60)

matches = matcher.find_matches(test_abstract, test_industry, limit=10)

print(f"\nTop {len(matches)} Results:")
for i, match in enumerate(matches, 1):
    outlet = match['outlet']
    score = match['score']
    confidence = match['match_confidence']
    explanation = match['match_explanation']
    
    print(f"\n{i}. {outlet.get('Outlet Name', 'Unknown')}")
    print(f"   Final Score: {score:.3f} ({confidence})")
    print(f"   Audience: {outlet.get('Audience', 'N/A')}")
    print(f"   Keywords: {outlet.get('Keywords', 'N/A')}")
    print(f"   Explanation: {explanation}")
    print("-" * 40)

# Show what should be the expected results
print("\n🎯 EXPECTED RESULTS ANALYSIS:")
print("=" * 60)

education_outlets = []
tech_outlets = []
other_outlets = []

for outlet in outlets:
    name = outlet.get('Outlet Name', 'Unknown')
    audience = outlet.get('Audience', '').lower()
    keywords = outlet.get('Keywords', '').lower()
    
    if 'education' in audience or 'education' in keywords:
        education_outlets.append(outlet)
    elif 'tech' in audience or 'technology' in keywords or 'software' in keywords:
        tech_outlets.append(outlet)
    else:
        other_outlets.append(outlet)

print(f"Education-focused outlets: {len(education_outlets)}")
for outlet in education_outlets[:5]:
    print(f"  - {outlet.get('Outlet Name', 'Unknown')} (Audience: {outlet.get('Audience', 'N/A')})")

print(f"\nTechnology-focused outlets: {len(tech_outlets)}")
for outlet in tech_outlets[:5]:
    print(f"  - {outlet.get('Outlet Name', 'Unknown')} (Audience: {outlet.get('Audience', 'N/A')})")

print(f"\nOther outlets: {len(other_outlets)}") 