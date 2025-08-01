import os
import json
import re
import requests
import streamlit as st
from datetime import datetime
import google.generativeai as genai
import google.api_core.exceptions
import ast
import concurrent.futures
from functools import lru_cache
import random

# ---------- Config ----------
st.set_page_config(page_title="QLOOTRA ✈️", page_icon="🌍", layout="wide")

from plan_trip import (
    get_recommendations_with_batched_fallback,
    clean_items,
)

# ---------- Global cache for Gemini responses ----------
GEMINI_CACHE = {}

# ---------- Load API keys from Streamlit secrets ----------
gemini_keys = st.secrets["GEMINI_KEYS"].split(",")
gemini_key = random.choice(gemini_keys).strip()
genai.configure(api_key=gemini_key)


qloo_key = st.secrets["QLOO_API_KEY"]
if not qloo_key.startswith("Bearer "):
    qloo_key = "Bearer " + qloo_key




if not gemini_keys:
    st.error("❌ Gemini API keys missing in any of .gemini_key1 - .gemini_key10")
if not qloo_key:
    st.warning("⚠️ Qloo API key missing in .qloo_key")

# MODIFICATION: Global variable to track current Gemini key index
if "current_gemini_key_idx" not in st.session_state:
    st.session_state.current_gemini_key_idx = 0

# MODIFICATION: Define generate_with_gemini helper function with key rotation
def generate_with_gemini(prompt):
    """Generate response using Gemini with key rotation and error handling."""
    if not gemini_keys:
        return "❌ No Gemini API keys configured."

    # Try up to 2 keys (current and next) before giving up
    for attempt in range(2):
        try:
            genai.configure(api_key=gemini_keys[st.session_state.current_gemini_key_idx])
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            response = model.generate_content(prompt, generation_config=genai.GenerationConfig(max_output_tokens=2000))
            return response.text
        except google.api_core.exceptions.ResourceExhausted:
            # Rate limit hit, rotate key
            st.session_state.current_gemini_key_idx = (st.session_state.current_gemini_key_idx + 1) % len(gemini_keys)
            st.warning(f"⚠️ Rotating to Gemini API key #{st.session_state.current_gemini_key_idx + 1}")
        except Exception as e:
            # Other error, log and try next key
            st.error(f"❌ Gemini error with key #{st.session_state.current_gemini_key_idx + 1}: {e}")
            st.session_state.current_gemini_key_idx = (st.session_state.current_gemini_key_idx + 1) % len(gemini_keys)

    return "❌ All Gemini API keys exhausted or failed."


def get_batched_gemini_recommendations(taste_domain_pairs):
    """
    Batch multiple taste-domain queries into a single Gemini API call.
    Input: List of (taste, domain) tuples
    Returns: Dict of {(taste, domain): [items]}
    """
    # Check cache first
    cached_results = {}
    remaining_pairs = []
    
    for pair in taste_domain_pairs:
        if pair in GEMINI_CACHE:
            cached_results[pair] = GEMINI_CACHE[pair]
        else:
            remaining_pairs.append(pair)
    
    if not remaining_pairs:
        return cached_results
    
    # Create a single prompt for all remaining pairs
    prompt_parts = [
        "Please provide recommendations for the following preferences. ",
        "For each line, respond with 'taste:domain: item1, item2, item3'\n"
    ]
    
    for taste, domain in remaining_pairs:
        prompt_parts.append(f"{taste}:{domain}: ")
    
    prompt = "\n".join(prompt_parts)
    
    try:
        gemini_response = generate_with_gemini(prompt)
        
        # Parse the response
        new_results = {}
        for line in gemini_response.split("\n"):
            if ":" in line:
                parts = line.split(":", 2)  # Split into at most 3 parts
                if len(parts) >= 3:
                    taste_part = parts[0].strip()
                    domain_part = parts[1].strip()
                    items_str = parts[2].strip()
                    
                    # Find matching pair
                    for taste, domain in remaining_pairs:
                        if taste_part.lower() == taste.lower() and domain_part.lower() == domain.lower():
                            items = [i.strip() for i in items_str.split(",") if i.strip()]
                            new_results[(taste, domain)] = items
                            GEMINI_CACHE[(taste, domain)] = items
                            break
        
        # Combine cached and new results
        cached_results.update(new_results)
        return cached_results
        
    except Exception as e:
        st.error(f"❌ Error in batched Gemini request: {e}")
        # Fallback to individual requests
        results = cached_results.copy()
        for taste, domain in remaining_pairs:
            try:
                prompt = f"A person likes '{taste}'. Suggest up to 3 popular {domain} items for a traveler. Respond as: {domain}: item1, item2, item3"
                response = generate_with_gemini(prompt)
                
                # Parse response
                items = []
                for line in response.split("\n"):
                    if line.strip().startswith(f"{domain}:"):
                        items_str = line.split(":", 1)[1].strip()
                        items = [i.strip() for i in items_str.split(",") if i.strip()]
                        break
                
                results[(taste, domain)] = items[:3]
                GEMINI_CACHE[(taste, domain)] = items[:3]
            except Exception as e:
                st.error(f"❌ Error getting individual Gemini recommendation for {taste}/{domain}: {e}")
                results[(taste, domain)] = []
        
        return results


# Initial model configuration (uses the first key initially)
if gemini_keys:
    genai.configure(api_key=gemini_keys[st.session_state.current_gemini_key_idx])
    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)
else:
    model = None # Handle case where no keys are loaded


# ---------- Paths & Domains ----------
TRIP_FILE = "trips.json"
TASTE_FILE = "tastes.json"
QLOO_DOMAINS = {
    "music": "music",
    "food": "food",
    "fashion": "fashion",
    "tv": "tv",
    "movie": "movies",
    "brand": "brands",
    "podcast": "podcasts",
    "book": "books",
    "game": "games",
    "travel": "place",
    "place": "place",
}

# ---------- Load/Save ----------
@st.cache_data(ttl=3600)
def load_saved_tastes():
    if os.path.exists(TASTE_FILE):
        try:
            with open(TASTE_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_tastes(tastes):
    with open(TASTE_FILE, "w") as f:
        json.dump(tastes, f, indent=2)

@st.cache_data(ttl=3600)
def load_saved_trips():
    if os.path.exists(TRIP_FILE):
        try:
            with open(TRIP_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_trips(trips):
    with open(TRIP_FILE, "w") as f:
        json.dump(trips, f, indent=2)

# ---------- State Init ----------
def init_state():
    defaults = {
        "mode": "Normal",
        "chat_normal": [],
        "chat_trip": [],
        "chat_spark": [],
        "greeted_normal": False,
        "greeted_trip": False,
        "greeted_spark": False,
        "tastes": load_saved_tastes(),
        "trip": {},
        "spark_summary": "",
        "trip_phase": "PLAN",
        "location": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------- Utilities ----------
def detect_location():
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5)
        if r.status_code == 200:
            d = r.json()
            return ", ".join(filter(None, [d.get("city"), d.get("region"), d.get("country")]))
    except:
        pass
    return "Unknown"

if not st.session_state.location:
    st.session_state.location = detect_location()

def clean_json_from_text(text):
    """Get pure JSON substring from Gemini response (regex fallback)"""
    text = text.strip()
    # Remove markdown triple backticks and language specifier (e.g. ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip('` \n')
    # Try to find the first JSON object block
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

@lru_cache(maxsize=128) # Cache Gemini text extraction
def extract_tastes_from_text(text):
    prompt = (
        "Extract all user tastes related to music artists, food items, fashion, movies, TV shows, brands, "
        "books, locations, or hobbies from the following input. Output only a Python list of entities. "
        "Example: ['Jazz Music', 'Italian Food', 'Hiking']\n" # Added example for clarity
        f'Input: "{text}"'
    )
    try:
        resp_text = generate_with_gemini(prompt)
        cleaned = resp_text.strip("` \n")
        try:
            parsed = json.loads(cleaned)
        except Exception:
            try:
                parsed = ast.literal_eval(cleaned)
            except:
                parsed = []
        if isinstance(parsed, list):
            # Return tuple for lru_cache hashability
            return tuple(sorted([str(e).strip() for e in parsed if e]))
        return tuple()
    except Exception as e:
        st.error(f"Error extracting tastes: {e}")
        return tuple()

def qloo_request_get(url, params=None):
    headers = {"X-Api-Key": qloo_key}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.warning(f"Qloo API error: {resp.status_code} - {resp.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Qloo Request Error: {e}")
        return None

# MODIFICATION 1: Parallelize Qloo API Requests (with fixed URL)
def get_qloo_cross_domain_recs_parallel(taste, max_per_domain=3):
    if not qloo_key or not taste:
        return {}
    base_url = "https://hackathon.api.qloo.com/v2/recommendations/"
    recommendations = {}

    def fetch(domain):
        url = f"{base_url}{QLOO_DOMAINS[domain]}"
        params = {"entity": taste}
        data = qloo_request_get(url, params=params)
        if data and domain in data:
            recs = data.get(domain, [])
            names = [item.get("name", "") for item in recs[:max_per_domain] if item.get("name")]
            return domain, tuple(sorted(names)) # Return tuple for consistency with caching
        else:
            return domain, tuple()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(QLOO_DOMAINS)) as executor:
        future_to_domain = {executor.submit(fetch, domain): domain for domain in QLOO_DOMAINS.keys()}
        for future in concurrent.futures.as_completed(future_to_domain):
            domain, names = future.result()
            recommendations[domain] = names
    return recommendations

# MODIFICATION 2: Cache Qloo (and Gemini) Results Per-Taste
@lru_cache(maxsize=128)
def cached_qloo_recs(taste):
    return get_qloo_cross_domain_recs_parallel(taste)

@lru_cache(maxsize=128) # Cache Gemini fallback results
def cached_gemini_fallback(domain, taste): # Added 'taste' parameter for specific context
    prompt = (
        f"A person likes '{taste}'. Suggest 3 popular {domain} they might also enjoy. "
        "Output as a comma-separated list only."
    )
    response = generate_with_gemini(prompt)
    items = tuple(sorted([item.strip(" .") for item in response.split(",") if item.strip()])) # Return tuple for lru_cache
    return items if items else tuple()
def threaded_get_recs(taste, domains=None):
    if not taste:
        return {}
    if domains is None:
        domains = list(QLOO_DOMAINS.keys())

    results = {}

    def fetch(domain):
        qloo_items = list(cached_qloo_recs(taste).get(domain, []))
        if len(qloo_items) < 2:
            fallback_items = list(cached_gemini_fallback(domain, taste))[:3]
            qloo_items.extend(fallback_items)
        return domain, list(set(qloo_items))[:5]  # deduplicate & limit

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch, domain) for domain in domains]
        for future in concurrent.futures.as_completed(futures):
            domain, items = future.result()
            results[domain] = items

    return results

# MODIFICATION 3: Optimize Fallback Logic
def get_recommendations_with_fallback(taste):
    if not taste:
        return {}

    final_recs = {}

    # Try cached Qloo recommendations first
    qloo_data = cached_qloo_recs(taste) or {}

    for domain in QLOO_DOMAINS.keys():
        # Get Qloo items for domain
        qloo_items = list(qloo_data.get(domain, []))
        qloo_items = [item.strip() for item in qloo_items if item]

        # Initialize final recs with Qloo items
        final_recs[domain] = qloo_items

        # Fallback to Gemini only if Qloo has less than 2 items
        if len(qloo_items) < 2:
            fallback_items = cached_gemini_fallback(domain, taste) or []
            fallback_items = [item.strip() for item in fallback_items if item]

            # Add fallback items (max 3), deduplicated
            final_recs[domain].extend(fallback_items[:3])

        # Final cleanup: deduplicate and limit to 5 items
        seen = set()
        cleaned = []
        for item in final_recs[domain]:
            if item.lower() not in seen:
                seen.add(item.lower())
                cleaned.append(item)
        final_recs[domain] = cleaned[:5]

    return final_recs


@lru_cache(maxsize=128) # Cache friendly responses as well
def generate_friendly_response(prompt):
    try:
        return generate_with_gemini(prompt)
    except Exception as e:
        st.error(str(e))
        return "Sorry, I'm having trouble answering right now."

# MODIFICATION 5: Batch/Reduce Gemini Calls in generate_chat_reply
def generate_chat_reply(user_msg):
    tastes_extracted = extract_tastes_from_text(user_msg) # This is already cached
    new_tastes = []
    for t in tastes_extracted:
        if t.lower() not in (x.lower() for x in st.session_state.tastes):
            st.session_state.tastes.append(t)
            new_tastes.append(t)
    
    # Collect all recommendations first
    all_recs_details = []
    if new_tastes:
        for taste in new_tastes:
            recs_for_taste = get_recommendations_with_fallback(taste) # Uses optimized caching and fallback
            rec_strings_for_taste = [
                f"{', '.join(items)} ({dom})" for dom, items in recs_for_taste.items() if items
            ]
            if rec_strings_for_taste:
                all_recs_details.append(f"For {taste}, try: {', '.join(rec_strings_for_taste)}.")
    
    combined_recs_text = " ".join(all_recs_details)

    # Call Gemini only once with a comprehensive prompt
    if new_tastes:
        chat_prompt = (
            f"You are a witty travel chatbot. The user said: '{user_msg}'. "
            f"You detected these new tastes: {', '.join(new_tastes)}. " # Join new_tastes for better prompt
            f"Here are some recommendations based on these: {combined_recs_text} "
            "Make your reply natural and engaging, not too formal! Incorporate the recommendations smoothly."
        )
    else:
        chat_prompt = (
            f"User: '{user_msg}'. Reply warmly and informatively, even if you do not detect a clear taste. "
            f"Current location: {st.session_state.location}. Ask them if they'd like to plan a trip!"
        )
    
    reply = generate_friendly_response(chat_prompt)
    return reply

# ---------- Sidebar ----------
with st.sidebar:
    st.header("QLOOTRA Modes")
    # Fixed: Changed key from "main" to "mode" to match the session state checks
    st.radio("Choose a Mode", ("Normal", "Plan a Trip", "Spark"), key="mode")
    st.markdown("---")
    st.text_input("📍 Your Location", key="location")
    st.markdown("🎵 **Tastes Memory**:")
    st.text_input("Music", value=st.session_state.get("tastes_music", ""), key="tastes_music")
    st.text_input("Food", value=st.session_state.get("tastes_food", ""), key="tastes_food")
    st.text_input("Fashion", value=st.session_state.get("tastes_fashion", ""), key="tastes_fashion")

    if st.button("🔁 Discover My Identity"):
        mus = st.session_state.tastes_music
        if mus:
            with st.spinner("Analyzing your cultural identity..."):
                summary = generate_with_gemini(
                    f"Write a fun 1-line summary of a person who likes {mus}. No explanations."
                )
            if mus not in st.session_state.tastes:
                if mus.lower() not in [t.lower() for t in st.session_state.tastes]:
                    st.session_state.tastes.append(mus)
            save_tastes(st.session_state.tastes)
            st.session_state.chat_normal.append(("assistant", summary))
            st.toast("Cultural identity generated! 🌟")
        else:
            st.warning("Enter a music taste first.")
    
    st.markdown("---")
    if st.session_state.tastes:
        st.markdown("**Current Tastes:**")
        for t in st.session_state.tastes:
            st.write(f"- {t}")
    else:
        st.write("No tastes yet. Mention some music, food, or fashion!")

    # --- Trip Save/Show ---
    past_trips = load_saved_trips()
    st.markdown("---")
    st.header("🗂️ Past Trips")
    if past_trips:
        for trip in reversed(past_trips[-5:]):
            st.markdown(
                f"- **{trip.get('destination', 'Unknown')}**, {trip.get('days', '?')} days, "
                f"Budget: {trip.get('budget', 'N/A')}"
            )
    else:
        st.write("No trips saved yet.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        for key in ("chat_normal", "chat_trip", "chat_spark"):
            st.session_state[key] = []
        for key in ("greeted_normal", "greeted_trip", "greeted_spark"):
            st.session_state[key] = False
        
    if st.button("🧠 Forget Tastes"):
        st.session_state.tastes = []
        save_tastes([])
        st.toast("Taste memory cleared!")
         # Ensure immediate update

# ---------- Plan a Trip Button ----------
st.markdown(
    '<div style="text-align:center;"><button onclick="window.location.reload();" '
    'style="background:#17d4aa;color:white;font-weight:bold;font-size:18px;'
    'padding:10px 40px;border:none;border-radius:30px;box-shadow:0 2px 6px #aaa;'
    'cursor:pointer;">✈️ Plan a Trip</button></div>',
    unsafe_allow_html=True
)

# --- Helper for cleaning items (deduplication and stripping) ---
def clean_items(items):
    cleaned = set()
    for i in items:
        i = str(i).strip() # Ensure item is string before stripping
        if i:
            cleaned.add(i)
    return sorted(list(cleaned)) # Return sorted list for consistent display

# New function for building paragraph summaries
def build_paragraphs_for_spark_summary(destination, days, all_recs_raw):
    # Clean and gather items
    def clean_items_local(items):
        return sorted(list(set(str(i).strip() for i in items if str(i).strip())))
    
    # PACK phase (fashion + brands)
    pack_items = clean_items_local(all_recs_raw.get("fashion", []))[:days]
    brands = clean_items_local(all_recs_raw.get("brand", []))[:5]
    pack_paragraph = ""
    if pack_items or brands:
        pack_paragraph = (
            "For your packing essentials, consider stylish and comfortable outfits like "
            f"{', '.join(pack_items) if pack_items else 'versatile attire'}. " # Added fallback for join
            "Pair these with trusted brands such as "
            f"{', '.join(brands) if brands else 'your favorite comfortable brands'}. " # Added fallback for join
            "These will keep you both fashionable and prepared during your trip."
        )
    else:
        pack_paragraph = "We couldn't find specific packing recommendations, but packing versatile and trusty essentials is always a good choice."

    # JOURNEY phase (entertainment + foods)
    journey_domains = ["music", "movie", "tv", "podcast", "book", "game"]
    entertainment = set()
    for dom in journey_domains:
        entertainment.update(all_recs_raw.get(dom, []))
    entertainment = clean_items_local(entertainment)[:10]
    journey_foods = clean_items_local(all_recs_raw.get("food", []))[:5]
    journey_paragraph = ""
    if entertainment or journey_foods:
        journey_paragraph = (
            "During your journey, enjoy some fantastic entertainment options such as "
            f"{', '.join(entertainment)}"
            if entertainment else "no specific entertainment recommendations"
        )
        if journey_foods:
            journey_paragraph += (
                f". And to delight your taste buds on the go, try snacks like {', '.join(journey_foods)}."
            )
    else:
        journey_paragraph = "We recommend bringing your favorite entertainment and snacks to enjoy while traveling."

    # DESTINY phase (local foods and places)
    destiny_places = clean_items_local(
        list(all_recs_raw.get("travel", [])) + list(all_recs_raw.get("place", []))
    )
    if destination and destination not in destiny_places:
        destiny_places.insert(0, destination)
    destiny_places = destiny_places[:7]
    destiny_foods = clean_items_local(all_recs_raw.get("food", []))[:5]
    destiny_paragraph = ""
    if destiny_places or destiny_foods:
        destiny_paragraph = (
            "Once you arrive, don’t miss trying some local delicacies such as "
            f"{', '.join(destiny_foods) if destiny_foods else 'local favorites'}. "
            "Also, be sure to visit iconic spots like "
            f"{', '.join(destiny_places) if destiny_places else 'the main attractions'}. "
            "Experiencing these will enrich your travel adventure."
        )
    else:
        destiny_paragraph = "Explore the local culture and cuisine to immerse yourself in the destination."

    # RETURN phase (reflection)
    return_paragraph = "At the end of your trip, take some time to reflect and share your experience and memories."

    # Compose full summary
    summary = (
        f"**PLAN:** Prepare for your {days}-day trip to {destination}.\n\n"
        f"**PACK:** {pack_paragraph}\n\n"
        f"**JOURNEY:** {journey_paragraph}\n\n"
        f"**DESTINY:** {destiny_paragraph}\n\n"
        f"**RETURN:** {return_paragraph}\n"
    )
    return summary

# ---------- MAIN MODES ----------
if st.session_state.mode == "Normal":
    st.subheader("🧠 Normal Chat")

    # Initialize session state variables
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "greeted" not in st.session_state:
        st.session_state.greeted = False
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # Greet the user once on first load (no rerun here, just append)
    if not st.session_state.greeted:
        greeting_text = f"Hi! I'm QLOOTRA, your personal assistant. I'm currently in {st.session_state.location}. Tell me about your tastes or ask me anything!"
        st.session_state.chat.append(("assistant", greeting_text))
        st.session_state.greeted = True

    # Display all chat messages
    for sender, message in st.session_state.chat:
        role = "user" if sender == "user" else "assistant"
        st.chat_message(role).markdown(message)

    # User input box
    user_message = st.chat_input("Say something...")

    if user_message:
        # Append user message immediately so visible right away
        st.session_state.chat.append(("user", user_message))
        # Save as pending input to trigger assistant reply generation
        st.session_state.pending_input = user_message
        # Rerun to show user message immediately, assistant reply generation will happen next run
        st.rerun()

    # If there is pending user input without reply, generate assistant reply here
    if st.session_state.pending_input is not None:
        with st.spinner("Thinking and fetching recommendations..."):
            ai_reply = generate_chat_reply(st.session_state.pending_input)
        st.session_state.chat.append(("assistant", ai_reply))
        save_tastes(st.session_state.tastes)
        st.session_state.pending_input = None  # Reset to avoid duplicate replies
        st.rerun()


        # Ensure immediate update after user input and reply
# Store functions in session_state to share with plan_trip.py
# ---------- Store shared functions in session_state BEFORE importing plan_trip ----------
import streamlit as st
from datetime import datetime
import concurrent.futures
import re

# --- Helper functions for output filtering ---

def is_url(item):
    """
    Detect if the string looks like a URL or contains domain-like substrings.
    """
    url_pattern = re.compile(r'https?://|www\.|\.com|\.net|\.org', re.IGNORECASE)
    return bool(url_pattern.search(item))

def filter_items(items):
    """
    Filter out URLs, generic placeholders, and unrelated terms from recommendation lists.
    """
    exclude_set = {
        'netflix.com',
        'www.spotify.com',
        'spotify',
        'apple music',
        'outfits',
        'movies',
        'portable charger',
        'downloaded album',
        'playlist',
        'travel soundtrack',
        # add more words to exclude as needed
    }
    filtered = []
    for i in items:
        item_lower = i.lower()
        if not is_url(i) and all(excl not in item_lower for excl in exclude_set):
            filtered.append(i)
    return filtered

# --- Wrapper functions that call st.session_state backend functions ---
def call_generate_with_gemini(prompt):
    return st.session_state.generate_with_gemini_func(prompt)

def call_cached_qloo_recs(taste):
    if "cached_qloo_recs_func" not in st.session_state:
        st.error("❌ cached_qloo_recs_func not initialized.")
        return {}
    return st.session_state.cached_qloo_recs_func(taste)

def call_extract_tastes_from_text(text):
    return st.session_state.extract_tastes_from_text_func(text)

def call_load_saved_trips():
    return st.session_state.load_saved_trips_func()

def call_save_trips(trips):
    return st.session_state.save_trips_func(trips)

# --- Utility helper ---
def clean_items(items):
    seen = set()
    cleaned = []
    for i in items:
        s = str(i).strip()
        if s and s.lower() not in seen:
            cleaned.append(s)
            seen.add(s.lower())
    return cleaned

def get_qloo_recs_threaded(taste, domains):
    results = {}

    def fetch(domain):
        return domain, list(call_cached_qloo_recs(taste).get(domain, []))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(domains)) as executor:
        futures = [executor.submit(fetch, domain) for domain in domains]
        for future in concurrent.futures.as_completed(futures):
            domain, items = future.result()
            results[domain] = items or []
    return results

def get_batched_gemini_recommendations(taste_domain_pairs):
    results = {}
    taste_to_domains = {}
    for taste, domain in taste_domain_pairs:
        taste_to_domains.setdefault(taste, []).append(domain)

    for taste, domains in taste_to_domains.items():
        prompt = (
            f"A person likes '{taste}'. Suggest up to 3 popular {', '.join(domains)} items for a traveler.\n"
            + "\n".join([f"{dom}: item1, item2, item3" for dom in domains])
            + "\n\nOnly respond with domain names followed by items separated by commas."
        )
        gemini_response = call_generate_with_gemini(prompt)
        for line in gemini_response.splitlines():
            if ':' in line:
                dom, items_str = line.split(':', 1)
                dom = dom.strip().lower()
                if dom in domains:
                    items = [i.strip() for i in items_str.split(',') if i.strip()]
                    results[(taste, dom)] = items[:5]
    return results

def get_recommendations_with_batched_fallback(taste, needed_domains=None):
    if not taste:
        return {}

    if needed_domains is None:
        needed_domains = ["music", "food", "fashion", "movie", "travel", "place"]

    qloo_recs = get_qloo_recs_threaded(taste, needed_domains)
    final_recs = {}
    missing_domains = []

    for domain in needed_domains:
        items = qloo_recs.get(domain, [])
        cleaned = clean_items(items)[:5]
        final_recs[domain] = cleaned
        if len(cleaned) < 2:
            missing_domains.append(domain)

    if missing_domains:
        pairs = [(taste, domain) for domain in missing_domains]
        gemini_results = get_batched_gemini_recommendations(pairs)
        for (taste_key, domain), items in gemini_results.items():
            if domain in missing_domains:
                merged = clean_items(final_recs.get(domain, []) + items)[:5]
                final_recs[domain] = merged

    return final_recs

@st.cache_data(show_spinner=False)
def cached_recs_for_taste(taste):
    return get_recommendations_with_batched_fallback(taste)


def plan_trip_mode():
    st.subheader("✈️ Plan a Trip (Game Mode)")

    if "chat_trip" not in st.session_state:
        st.session_state.chat_trip = []
    if "greeted_trip" not in st.session_state:
        st.session_state.greeted_trip = False
    if "greeted_trip_done" not in st.session_state:
        st.session_state.greeted_trip_done = False
    if "trip" not in st.session_state:
        st.session_state.trip = {}
    if "tastes" not in st.session_state:
        st.session_state.tastes = []

    trip_phase = st.radio(
        "Which phase are you in?",
        ("PLAN", "PACK", "JOURNEY", "DESTINY", "RETURN"),
        key="trip_phase",
        horizontal=True,
    )

    # Show welcome once
    if not st.session_state.greeted_trip_done:
        if not st.session_state.greeted_trip:
            st.session_state.chat_trip.append((
                "assistant",
                "Welcome! Progress through the phases: PLAN, PACK, JOURNEY, DESTINY, RETURN — each step adapts based on your tastes."
            ))
            st.session_state.greeted_trip = True
        st.session_state.greeted_trip_done = True

    # Show chat history
    for sender, msg in st.session_state.chat_trip:
        role = "user" if sender == "user" else "assistant"
        st.chat_message(role).markdown(msg)

    def gather_tastes():
        combined = st.session_state.tastes + [
            st.session_state.get("tastes_music", ""),
            st.session_state.get("tastes_food", ""),
            st.session_state.get("tastes_fashion", ""),
        ]
        filtered = list(set(filter(None, combined)))
        return [t for t in filtered if len(t) >= 3][:5]

    if trip_phase == "PLAN":
        dest = st.text_input("Destination", value=st.session_state.trip.get("destination", ""))
        days = st.number_input("Days", min_value=1, max_value=60, value=st.session_state.trip.get("days", 3))
        budget = st.text_input("Budget (optional)", value=st.session_state.trip.get("budget", ""))

        if st.button("🚀 Lock Plan"):
            st.session_state.trip = {
                "destination": dest,
                "days": days,
                "budget": budget,
                "timestamp": datetime.now().isoformat(),
            }
            trips = call_load_saved_trips()
            trips.append(st.session_state.trip)
            call_save_trips(trips)
            msg = f"Your trip to **{dest}** for **{days} days** is locked in! (Budget: {budget or 'N/A'})"
            st.session_state.chat_trip.append(("assistant", msg))
            st.success(msg)

    elif trip_phase == "PACK":
        rec_tastes = gather_tastes()
        if rec_tastes:
            selected_tastes = st.multiselect(
                "Select tastes to base packing list on:",
                options=rec_tastes,
                default=rec_tastes,
            )
        else:
            selected_tastes = []

        outfits = set()
        brands = set()
        day_count = st.session_state.trip.get("days", 3)
        max_outfits = min(day_count, 10)

        if st.button("Generate Packing List"):
            with st.spinner("Generating packing recommendations..."):
                for taste in selected_tastes[:5]:
                    recs = cached_recs_for_taste(taste)
                    outfits.update(recs.get("fashion", []))
                    brands.update(recs.get("brand", []))
                    if len(outfits) >= max_outfits:
                        break

            outfits_list = clean_items(list(outfits))
            outfits_list = filter_items(outfits_list)[:max_outfits]

            brands_list = clean_items(list(brands))
            brands_list = filter_items(brands_list)[:5]

            msg = ""
            if outfits_list:
                msg += f"🧳 Pack these outfits: {', '.join(outfits_list)}\n"
            if brands_list:
                msg += f"🏷️ Brands to try: {', '.join(brands_list)}"
            if not msg.strip():
                msg = "No fashion/brand suggestions yet. Add more tastes!"
            st.session_state.chat_trip.append(("assistant", msg))
            st.markdown(msg)

    elif trip_phase == "JOURNEY":
        rec_tastes = gather_tastes()
        if rec_tastes:
            selected_tastes = st.multiselect(
                "Select tastes to base journey entertainment and snacks on:",
                options=rec_tastes,
                default=rec_tastes,
            )
        else:
            selected_tastes = []

        entertainment_items = set()
        journey_foods_items = set()
        journey_domains = ["music", "tv", "movie", "podcast", "book", "game"]

        if st.button("Generate Journey Entertainment & Snacks"):
            with st.spinner("Finding travel entertainment and snacks..."):
                for taste in selected_tastes[:5]:
                    recs = cached_recs_for_taste(taste)
                    for dom in journey_domains:
                        entertainment_items.update(recs.get(dom, []))
                    journey_foods_items.update(recs.get("food", []))

            entertainment = clean_items(list(entertainment_items))
            entertainment = filter_items(entertainment)[:10]

            journey_foods = clean_items(list(journey_foods_items))
            journey_foods = filter_items(journey_foods)[:5]

            msg = ""
            if entertainment:
                msg += f"🎬 Entertainment: {', '.join(entertainment)}\n"
            if journey_foods:
                msg += f"🍔 Travel foods: {', '.join(journey_foods)}"
            if not msg.strip():
                msg = "No travel recommendations yet. Add your tastes to get suggestions!"
            st.session_state.chat_trip.append(("assistant", msg))
            st.markdown(msg)

    elif trip_phase == "DESTINY":
        dest = st.session_state.trip.get("destination", "")
        taste_seeds = list(set(filter(None, st.session_state.tastes + [
            st.session_state.get("tastes_music", ""),
            st.session_state.get("tastes_food", ""),
        ])))

        if taste_seeds:
            selected_tastes = st.multiselect(
                f"Select tastes to discover highlights for {dest}:",
                options=taste_seeds,
                default=taste_seeds,
            )
        else:
            selected_tastes = []

        destiny_foods_items = set()
        destiny_places_items = set()

        if st.button(f"Discover {dest} Highlights"):
            with st.spinner(f"Discovering {dest}'s highlights..."):
                for taste in selected_tastes[:5]:
                    recs = cached_recs_for_taste(taste)
                    destiny_foods_items.update(recs.get("food", []))
                    destiny_places_items.update(recs.get("travel", []))
                    destiny_places_items.update(recs.get("place", []))

            destiny_foods = clean_items(list(destiny_foods_items))
            destiny_foods = filter_items(destiny_foods)[:5]

            shown_places = clean_items(list(destiny_places_items))
            shown_places = filter_items(shown_places)[:5]

            if dest and dest not in shown_places:
                shown_places.insert(0, dest)
                shown_places = shown_places[:5]

            msg = ""
            if destiny_foods:
                msg += f"🍲 Try these local foods: {', '.join(destiny_foods)}\n"
            if shown_places:
                msg += f"🗺️ Must-visit: {', '.join(shown_places)}"
            if not msg.strip():
                msg = "Tell me a bit more about your food or place tastes!"
            st.session_state.chat_trip.append(("assistant", msg))
            st.markdown(msg)

    elif trip_phase == "RETURN":
        rating = st.slider("How was your trip?", 1, 10, key="trip_rating")
        feedback = st.text_area("Feedback", key="trip_feedback")
        if st.button("Submit Feedback", key="trip_submit"):
            thank = "Thanks for your feedback! Glad to have travelled with you. 🚀"
            st.session_state.chat_trip.append(("assistant", thank))
            st.success(thank)

    trip_input = st.chat_input("Chat during your trip...", key="trip_input")
    if trip_input:
        st.session_state.chat_trip.append(("user", trip_input))
        with st.spinner("Processing your travel thoughts..."):
            tastes = call_extract_tastes_from_text(trip_input)
            rec_reply = ""
            for t in tastes:
                cr = cached_recs_for_taste(t)
                if cr:
                    rec_reply += f"🌍 Tips for **{t}**: "
                    for dom, items in cr.items():
                        clean_list = filter_items(clean_items(items))
                        rec_reply += f"{dom.title()}: {', '.join(clean_list)}; "
                else:
                    rec_reply += f"I'll remember {t} for the next phases! "
            if not rec_reply:
                rec_reply = "Enjoy your trip!"
        st.session_state.chat_trip.append(("assistant", rec_reply))
        st.chat_message("assistant").markdown(rec_reply)

# Caller example (if mode is set somewhere else in your app)
if st.session_state.get("mode") == "Plan a Trip":
    plan_trip_mode()



elif st.session_state.mode == "Spark":
    st.subheader("🌟 QLOOTRA SPARK MODE")

    # Initialize phase caches if not present
    if "phase_pack_cache" not in st.session_state:
        st.session_state.phase_pack_cache = {}
    if "phase_journey_cache" not in st.session_state:
        st.session_state.phase_journey_cache = {}
    if "phase_destiny_cache" not in st.session_state:
        st.session_state.phase_destiny_cache = {}

    def _spark_extract_trip_details(user_input):
        """Robustly extract destination, days, tastes from a 1-line trip prompt using Gemini."""
        prompt = f"""
You are a travel planner assistant. The user will give you a one-line trip description like:
"I'm going to Tokyo for 6 days, I like Justin Bieber."


Extract the following into JSON:
- "destination": the travel location
- "days": number of days as an integer
- "tastes": list of their interests (e.g. artist, food, movie, character)


Always return valid JSON only, nothing else. Like:
{{
  "destination": "Tokyo",
  "days": 6,
  "tastes": ["Justin Bieber"]
}}


User Input:
"{user_input}"
"""
        try:
            resp_text = generate_with_gemini(prompt)
            raw = clean_json_from_text(resp_text)
            trip_data = json.loads(raw)

            destination = str(trip_data.get("destination", "")).strip()
            try:
                days = int(trip_data.get("days", 3))
            except:
                days = 3

            tastes = trip_data.get("tastes", [])
            if isinstance(tastes, str):
                tastes = [tastes]
            tastes = [str(t).strip() for t in tastes if t]

            if not destination or not tastes:
                return None

            return {"destination": destination, "days": days, "tastes": tastes}

        except Exception as e:
            st.error(f"[Parsing Error] Could not process Gemini output: {e}")
            return None

    def _spark_get_qloo_insight(entity_name, domain_type="fashion"):
        try:
            search_url = f"https://hackathon.api.qloo.com/v2/entities/search?name={entity_name}"
            headers = {"X-Api-Key": qloo_key}
            search_resp = requests.get(search_url, headers=headers, timeout=8)
            data = search_resp.json()
            if not data.get("data"):
                return None
            entity_id = data["data"][0]["id"]
            insight_url = (
                f"https://hackathon.api.qloo.com/v2/insights/?filter.type=urn:entity:{domain_type}"
                f"&signal.interests.entities={entity_id}"
            )
            insight_resp = requests.get(insight_url, headers=headers, timeout=8)
            if insight_resp.status_code == 200:
                insights = insight_resp.json().get("data", [])
                if isinstance(insights, list) and insights:
                    return [str(i.get('name', '') or str(i))[:50] for i in insights[:5] if i.get("name")]
        except Exception:
            return None
        return None

    def _gemini_fallback_for_domain(taste, domain):
        prompt = f"Based on the user's interest in '{taste}', suggest something in the {domain} domain (food, fashion, movies, etc). Reply with 1-6 short names or items separated by commas."
        try:
            response_text = generate_with_gemini(prompt)
            # Split by comma, semicolon, or newline, take first 6
            suggestions = [s.strip() for s in re.split(r"[,;\n]", response_text) if s.strip()]
            return suggestions[:5] if suggestions else [f"(Gemini fallback failed for {domain})"]
        except Exception:
            return [f"(Gemini fallback failed for {domain})"]

    def _spark_generate_phase_suggestions(trip):
        destination = trip.get("destination", "your destination")
        days = trip.get("days", 3)
        tastes = trip.get("tastes", [])

        trip_key = f"{destination}_{days}"

        # Pack phase cache
        if trip_key in st.session_state.phase_pack_cache:
            pack = st.session_state.phase_pack_cache[trip_key]
        else:
            pack_set = set()
            for taste in tastes:
                fashion = _spark_get_qloo_insight(taste, "fashion")
                if not fashion:
                    fashion = _gemini_fallback_for_domain(taste, "fashion")
                elif isinstance(fashion, str):
                    fashion = [fashion]
                pack_set.update(fashion[:5])
            pack = sorted(pack_set) if pack_set else ["👕 Pack stylish and comfy outfits"]
            st.session_state.phase_pack_cache[trip_key] = pack

        # Journey phase cache
        if trip_key in st.session_state.phase_journey_cache:
            journey = st.session_state.phase_journey_cache[trip_key]
        else:
            journey_set = set()
            for taste in tastes:
                media = _spark_get_qloo_insight(taste, "movie")
                if not media:
                    media = _gemini_fallback_for_domain(taste, "movie")
                elif isinstance(media, str):
                    media = [media]
                journey_set.update(media[:5])
            journey = sorted(journey_set) if journey_set else ["🎬 Enjoy great movies and shows"]
            st.session_state.phase_journey_cache[trip_key] = journey

        # Destiny phase cache
        if trip_key in st.session_state.phase_destiny_cache:
            destiny = st.session_state.phase_destiny_cache[trip_key]
        else:
            destiny_set = set()
            for taste in tastes:
                food = _spark_get_qloo_insight(taste, "food")
                if not food:
                    food = _gemini_fallback_for_domain(taste, "food")
                elif isinstance(food, str):
                    food = [food]
                destiny_set.update(food[:5])
            destiny = sorted(destiny_set) if destiny_set else ["🍜 Try local flavors and hidden gems"]
            st.session_state.phase_destiny_cache[trip_key] = destiny

        return {
            "PLAN": f"🌍 You're going to **{destination}** for **{days}** days. Prepare for an adventure!",
            "PACK": [f"👕 {item}" for item in pack],
            "JOURNEY": [f"🎬 {item}" for item in journey],
            "DESTINY": [f"🍜 {item}" for item in destiny],
            "RETURN": "📝 At the end of your trip, rate your experience and share your memories!"
        }

    # --- SPARK UI ---
    spark_input = st.text_input(
        "Describe your trip in 1 line",
        placeholder="I'm going to Tokyo for 5 days. I like BTS and spicy food.",
        key="spark_input"
    )

    if spark_input:
        with st.spinner("Planning your journey..."):
            trip_data = _spark_extract_trip_details(spark_input)
            if not trip_data:
                st.error("⚠️ Couldn't parse input. Try: ‘I'm going to Goa for 3 days, I love Coldplay and seafood.’")
            else:
                st.markdown("#### 🧠 Parsed Trip Info")
                st.json(trip_data)

                full_plan = _spark_generate_phase_suggestions(trip_data)
                for phase, items in full_plan.items():
                    st.markdown(f"### {phase}")
                    if isinstance(items, list):
                        for item in items:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(items)





# ---------- Footer ----------
st.markdown("---")
st.caption("Built with ❤️ using Gemini + Qloo | QLOOTRA Hackathon")

# ---- Safe export for use in plan_trip.py via session_state (no circular import) ----
if "cached_qloo_recs_func" not in st.session_state:
    st.session_state.cached_qloo_recs_func = cached_qloo_recs

if "generate_with_gemini_func" not in st.session_state:
    st.session_state.generate_with_gemini_func = generate_with_gemini

if "extract_tastes_from_text_func" not in st.session_state:
    st.session_state.extract_tastes_from_text_func = extract_tastes_from_text

if "load_saved_trips_func" not in st.session_state:
    st.session_state.load_saved_trips_func = load_saved_trips

if "save_trips_func" not in st.session_state:
    st.session_state.save_trips_func = save_trips

