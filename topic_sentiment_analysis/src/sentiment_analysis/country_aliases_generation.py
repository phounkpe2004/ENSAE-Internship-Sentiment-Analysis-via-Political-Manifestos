import requests

def generate_wikidata_country_dictionary():
    # URL de l'API SPARQL de Wikidata
    url = "https://query.wikidata.org/sparql"
    
    # Requête SPARQL magique : 
    # Récupère le nom en anglais, tous les alias (also known as) et les gentilés (demonyms) de chaque entité de type "pays" (Q6256)
    query = """
    SELECT ?countryLabel (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) (GROUP_CONCAT(DISTINCT ?demonym; separator="|") AS ?demonyms) WHERE {
      ?country wdt:P31 wd:Q6256. # Instance de "pays"
      
      # Récupération des alias (en anglais)
      OPTIONAL {
        ?country skos:altLabel ?alias.
        FILTER(LANG(?alias) = "en")
      }
      
      # Récupération des gentilés (en anglais)
      OPTIONAL {
        ?country wdt:P1549 ?demonym.
        FILTER(LANG(?demonym) = "en")
      }
      
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    GROUP BY ?countryLabel
    """
    
    headers = {
        'User-Agent': 'PoliticalManifestoResearchBot/1.0 (gbedossoujunior@gmail.com)',
        'Accept': 'application/json'
    }
    
    print("Connexion à Wikidata...")
    response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
    data = response.json()
    
    raw_dict = {}
    
    # 1. Extraction et regroupement des termes bruts
    for item in data['results']['bindings']:
        # Nom standard du pays (ex: "Russia")
        main_name = item['countryLabel']['value']
        
        # Ignorer les labels génériques ou bizarres
        if "republic" in main_name.lower() and len(main_name.split()) > 3:
            continue
            
        all_terms = {main_name.lower()}
        
        # Ajout des alias ("Soviet Union", "USSR", etc.)
        if 'aliases' in item and item['aliases']['value']:
            for alias in item['aliases']['value'].split('|'):
                if len(alias.strip()) < 4:
                    continue  # On passe à l'alias suivant, on ne l'ajoute pas
                all_terms.add(alias.lower())
                
        # Ajout des gentilés ("Russian", "Russians", etc.)
        if 'demonyms' in item and item['demonyms']['value']:
            for demonym in item['demonyms']['value'].split('|'):
                all_terms.add(demonym.lower())
                # Forcer le pluriel simple s'il n'y est pas
                if not demonym.endswith('s'):
                    all_terms.add(f"{demonym}s".lower())
        
        # On garde l'entité si elle a des termes associés
        key_label = main_name.upper().replace(" ", "_")
        raw_dict[key_label] = list(all_terms)

    # 2. Conversion au format "Token Patterns" pour spaCy
    spacy_country_aliases = {}
    
    for country_label, terms in raw_dict.items():
        spacy_country_aliases[country_label] = []
        for term in terms:
            # On découpe le terme par mot pour créer la liste de dictionnaires spaCy
            # Exemple: "soviet union" -> [{"LOWER": "soviet"}, {"LOWER": "union"}]
            token_pattern = [{"LOWER": word} for word in term.split()]
            if token_pattern: # Éviter les chaînes vides
                spacy_country_aliases[country_label].append(token_pattern)
                
    return spacy_country_aliases
