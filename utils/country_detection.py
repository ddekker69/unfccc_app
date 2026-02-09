# utils/country_detection.py

import pycountry
import re

# === Additional Variants ===
additional_variants = {
    "United States": "United States of America",
    "Russia": "Russian Federation",
    "South Korea": "Korea, Republic of",
    "North Korea": "Korea, Democratic People's Republic of",
    "Iran": "Iran, Islamic Republic of",
    "Syria": "Syrian Arab Republic",
    "Laos": "Lao People's Democratic Republic",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Bolivia": "Bolivia, Plurinational State of",
    "Tanzania": "Tanzania, United Republic of",
    "Moldova": "Moldova, Republic of",
    "Czechia": "Czech Republic",
    "European Union": "European Union",
    "European Community": "European Union"
}

# === Regionals ===
regional_actors = ["European Union", "European Community", "OECD", "African Group", "G77", "Alliance of Small Island States", "Least Developed Countries"]

# === Get country names ===
country_names = []
for country in pycountry.countries:
    country_names.append(country.name)
    if hasattr(country, 'official_name'):
        country_names.append(country.official_name)
country_names_extended = country_names + list(additional_variants.keys())

# === Extract country from title ===
def extract_country(title):
    if not isinstance(title, str):
        return None
    title = title.lower()
    for country in country_names_extended:
        pattern = r'\b' + re.escape(country.lower()) + r'(?=\b|[^a-z])'
        if re.search(pattern, title):
            return additional_variants.get(country, country)
    return None

# === ISO code conversion ===
def get_iso_alpha3(country_name):
    try:
        return pycountry.countries.lookup(additional_variants.get(country_name, country_name)).alpha_3
    except:
        return None

# === All known countries ===
def get_all_known_countries():
    return sorted(set(country_names_extended))

# === Check if regional ===
def is_regional_actor(name):
    return name in regional_actors
