import numpy as np
import pandas as pd
from src.config import MCMCConfig

# Codes de vote bruts -> à filtrer avant modélisation
VALID_VOTE_CODES = {1, 2, 3}  # Yes, Abstain, No -- on exclut 8 (absent) et 9 (non-membre)


HARDCODED_COUNTRY_MAP = {
    # Amperand / Alternative cleanups
    "ANTIGUA & BARBUDA": "ATG",
    "ANTIGUA AND BARBUDA": "ATG",
    "SAINT KITTS & NEVIS": "KNA",
    "SAINT KITTS AND NEVIS": "KNA",
    "ST. KITTS AND NEVIS": "KNA",
    "ST. KITTS & NEVIS": "KNA",
    "SAINT LUCIA": "LCA",
    "ST. LUCIA": "LCA",
    "SAINT VINCENT & THE GRENADINES": "VCT",
    "SAINT VINCENT AND THE GRENADINES": "VCT",
    "ST. VINCENT & THE GRENADINES": "VCT",
    "ST. VINCENT AND THE GRENADINES": "VCT",
    "TRINIDAD & TOBAGO": "TTO", 
    "TRINIDAD AND TOBAGO": "TTO",
    'BOLIVIA (PLURINATIONAL STATE OF)': "BOL",
    'VENEZUELA (BOLIVARIAN REPUBLIC OF)': "VEN",
    'REPUBLIC OF MOLDOVA': "MDA",
    'REPUBLIC OF KOREA': "KOR",
    'MICRONESIA (FEDERATED STATES OF)': "FSM",
    'IRAN (ISLAMIC REPUBLIC OF)': "IRN",
    "DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA": "PRK",

    # Formal names vs Short names
    "BOLIVIA": "BOL",
    "BOLIVIA, PLURINATIONAL STATE OF": "BOL",
    "BRUNEI": "BRN",
    "BRUNEI DARUSSALAM": "BRN",
    "CAPE VERDE": "CPV",
    "CABO VERDE": "CPV",
    "IVORY COAST": "CIV",
    "COTE D'IVOIRE": "CIV",
    "COTE D’IVOIRE": "CIV",
    "NORTH KOREA": "PRK",
    "KOREA, DEMOCRATIC PEOPLE'S REPUBLIC OF": "PRK",
    "SOUTH KOREA": "KOR",
    "KOREA, REPUBLIC OF": "KOR",
    "IRAN": "IRN",
    "IRAN, ISLAMIC REPUBLIC OF": "IRN",
    "LAOS": "LAO",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAO",
    "MICRONESIA": "FSM",
    "MICRONESIA, FEDERATED STATES OF": "FSM",
    "MOLDOVA": "MDA",
    "MOLDOVA, REPUBLIC OF": "MDA",
    "RUSSIA": "RUS",
    "RUSSIAN FEDERATION": "RUS",
    "TANZANIA": "TZA",
    "UNITED REPUBLIC OF TANZANIA": "TZA",
    "VENEZUELA": "VEN",
    "VENEZUELA, BOLIVARIAN REPUBLIC OF": "VEN",
    "VIETNAM": "VNM",
    "VIET NAM": "VNM",
    "AFGHANISTAN": "AFG",
    "ANGOLA": "AGO",
    "ALBANIA": "ALB",
    "ALGERIA": "ALG",
    "ANDORRA": "AND",
    "UNITED ARAB EMIRATES": "ARE",
    "ARGENTINA": "ARG",
    "ARMENIA": "ARM",
    "AUSTRALIA": "AUS",
    "AUSTRIA": "AUT",
    "AZERBAIJAN": "AZE",
    "BURUNDI": "BDI",
    "BELGIUM": "BEL",
    "BENIN": "BEN",
    "BURKINA FASO": "BFA",
    "BANGLADESH": "BGD",
    "BULGARIA": "BGR",
    "BAHAMAS": "BHS",
    "BAHRAIN": "BHR",
    "BHUTAN": "BHU",
    "BOSNIA AND HERZEGOVINA": "BIH",
    "BOSNIA & HERZEGOVINA": "BIH",
    "BELARUS": "BLR",
    "BELIZE": "BLZ",
    "BOTSWANA": "BWA",
    "BRAZIL": "BRA",
    "BARBADOS": "BRB",
    "CENTRAL AFRICAN REPUBLIC": "CAF",
    "CAMBODIA": "KHM",
    "CANADA": "CAN",
    "SWITZERLAND": "CHE",
    "CHILE": "CHL",
    "CHINA": "CHN",
    "CAMEROON": "CMR",
    "DEMOCRATIC REPUBLIC OF THE CONGO": "COD",
    "DRC": "COD",
    "CONGO": "COG",
    "COLOMBIA": "COL",
    "COMOROS": "COM",
    "COSTA RICA": "CRI",
    "CUBA": "CUB",
    "CYPRUS": "CYP",
    "CZECH REPUBLIC": "CZE",
    "CZECHIA": "CZE",
    "GERMANY": "DEU",
    "DJIBOUTI": "DJI",
    "DOMINICA": "DMA",
    "DENMARK": "DNK",
    "DOMINICAN REPUBLIC": "DOM",
    "ECUADOR": "ECU",
    "EGYPT": "EGY",
    "EQUATORIAL GUINEA": "GNQ",
    "ERITREA": "ERI",
    "SPAIN": "ESP",
    "ESTONIA": "EST",
    "ETHIOPIA": "ETH",
    "FIJI": "FJI",
    "FINLAND": "FIN",
    "FRANCE": "FRA",
    "GABON": "GAB",
    "GAMBIA": "GMB",
    "UNITED KINGDOM": "GBR",
    "GEORGIA": "GEO",
    "GHANA": "GHA",
    "GUINEA": "GIN",
    "GUINEA-BISSAU": "GNB",
    "GUINEA BISSAU": "GNB",
    "GREECE": "GRC",
    "GRENADA": "GRN",
    "GUATEMALA": "GTM",
    "GUYANA": "GUY",
    "HAITI": "HTI",
    "HONDURAS": "HND",
    "CROATIA": "HRV",
    "HUNGARY": "HUN",
    "ICELAND": "ISL",
    "INDONESIA": "IDN",
    "INDIA": "IND",
    "IRELAND": "IRL",
    "IRAQ": "IRQ",
    "ISRAEL": "ISR",
    "ITALY": "ITA",
    "JAMAICA": "JAM",
    "JORDAN": "JOR",
    "JAPAN": "JPN",
    "KAZAKHSTAN": "KAZ",
    "KENYA": "KEN",
    "KYRGYZSTAN": "KGZ",
    "KIRIBATI": "KIR",
    "KUWAIT": "KWT",
    "LEBANON": "LBN",
    "LIBERIA": "LBR",
    "LIBYA": "LBY",
    "LIECHTENSTEIN": "LIE",
    "SRI LANKA": "LKA",
    "LESOTHO": "LSO",
    "LITHUANIA": "LTU",
    "LUXEMBOURG": "LUX",
    "LATVIA": "LVA",
    "MOROCCO": "MAR",
    "MONACO": "MCO",
    "MADAGASCAR": "MDG",
    "MALDIVES": "MDV",
    "MEXICO": "MEX",
    "MARSHALL ISLANDS": "MHL",
    "NORTH MACEDONIA": "MKD",
    "MALI": "MLI",
    "MALTA": "MLT",
    "MYANMAR": "MMR",
    "MONTENEGRO": "MNE",
    "MONGOLIA": "MNG",
    "MOZAMBIQUE": "MOZ",
    "MAURITANIA": "MRT",
    "MAURITIUS": "MUS",
    "MALAWI": "MWI",
    "MALAYSIA": "MYS",
    "NAMIBIA": "NAM",
    "NIGER": "NER",
    "NIGERIA": "NGA",
    "NICARAGUA": "NIC",
    "NETHERLANDS": "NLD",
    "NORWAY": "NOR",
    "NEPAL": "NPL",
    "NAURU": "NRU",
    "NEW ZEALAND": "NZL",
    "OMAN": "OMN",
    "PAKISTAN": "PAK",
    "PANAMA": "PAN",
    "PARAGUAY": "PRY",
    "PERU": "PER",
    "PHILIPPINES": "PHL",
    "PALAU": "PLW",
    "PAPUA NEW GUINEA": "PNG",
    "POLAND": "POL",
    "PORTUGAL": "PRT",
    "QATAR": "QAT",
    "ROMANIA": "ROU",
    "RWANDA": "RWA",
    "SAUDI ARABIA": "SAU",
    "SUDAN": "SDN",
    "SENEGAL": "SEN",
    "SEYCHELLES": "SYC",
    "SINGAPORE": "SGP",
    "SOLOMON ISLANDS": "SLB",
    "SIERRA LEONE": "SLE",
    "SLOVENIA": "SVN",
    "EL SALVADOR": "SLV",
    "SAN MARINO": "SMR",
    "SOMALIA": "SOM",
    "SOUTH SUDAN": "SSD",
    "SAO TOME AND PRINCIPE": "STP",
    "SAO TOME & PRINCIPE": "STP",
    "SURINAME": "SUR",
    "SLOVAKIA": "SVK",
    "SWEDEN": "SWE",
    "ESWATINI": "SWZ",
    "SWAZILAND": "SWZ",
    "SYRIA": "SYR",
    "SYRIAN ARAB REPUBLIC": "SYR",
    "TAJIKISTAN": "TJK",
    "CHAD": "TCD",
    "TOGO": "TGO",
    "THAILAND": "THA",
    "TIMOR-LESTE": "TLS",
    "EAST TIMOR": "TLS",
    "TURKMENISTAN": "TKM",
    "TONGA": "TON",
    "TUNISIA": "TUN",
    "TURKEY": "TUR",
    "TURKIYE": "TUR",
    "TUVALU": "TUV",
    "TAIWAN": "TWN",
    "UGANDA": "UGA",
    "UKRAINE": "UKR",
    "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "URUGUAY": "URY",
    "UZBEKISTAN": "UZB",
    "VANUATU": "VUT",
    "SAMOA": "WSM",
    "YEMEN": "YEM",
    "SOUTH AFRICA": "ZAF",
    "ZAMBIA": "ZMB",
    "ZIMBABWE": "ZWE"
}


def load_raw_votes(
    cfg: MCMCConfig, verbose: bool = False, vote_type: str = "Important"
) -> pd.DataFrame:

    votes_path = cfg.raw_data_dir / "un_votes.csv"
    matcher_path = cfg.raw_data_dir / "country_codes.csv"

    raw = pd.read_csv(votes_path, index_col=0)
    matcher = pd.read_csv(matcher_path)
    filtered = raw[raw["vote"].isin(VALID_VOTE_CODES)].copy()
  
    if vote_type == "All":
        pass  # Keep all valid votes

    elif vote_type == "NoNukes":
        # Exclude nuclear votes (keep only nu == 0; drop NaNs and nu == 1)
        if "nu" in filtered.columns:
            filtered = filtered[filtered["nu"] == 0]

    elif vote_type == "Important":
        # Important votes only (session >= 38 and importantvote == 1)
        cond = filtered["session"] >= 38
        if "importantvote" in filtered.columns:
            cond &= filtered["importantvote"] == 1
        filtered = filtered[cond]

    elif vote_type == "MiddleEast":
        # Middle East votes (me == 1, session >= 29)
        filtered = filtered[(filtered["me"] == 1) & (filtered["session"] >= 29)]

    elif vote_type == "HumanRights":
        # Human Rights votes (hr == 1, session >= 25)
        filtered = filtered[(filtered["hr"] == 1) & (filtered["session"] >= 25)]

    elif vote_type == "Colonial":
        # Colonialism votes (co == 1, session >= 14)
        filtered = filtered[(filtered["co"] == 1) & (filtered["session"] >= 14)]

    elif vote_type == "Nuclear":
        # Nuclear votes (nu == 1, session >= 27)
        filtered = filtered[(filtered["nu"] == 1) & (filtered["session"] >= 27)]

    elif vote_type == "Economic":
        # Economic votes (ec == 1, session >= 26)
        filtered = filtered[(filtered["ec"] == 1) & (filtered["session"] >= 26)]

    elif vote_type == "Disarmament":
        # Disarmament votes (di == 1)
        filtered = filtered[filtered["di"] == 1]

    else:
        raise ValueError(
            f"Type de vote inconnu : '{vote_type}'. "
            "Options valides : 'All', 'NoNukes', 'Important', 'MiddleEast', "
            "'HumanRights', 'Colonial', 'Nuclear', 'Economic', 'Disarmament'"
        )

    # --- Verbose Output ---
    if verbose:
        print("Colonnes disponibles :", raw.columns.tolist())
        print(f"\nLignes avant filtrage : {len(raw)}")
        print(f"Lignes après filtrage ({vote_type}) : {len(filtered)}")
        print(f"Lignes retirées : {len(raw) - len(filtered)}")

        print("\nDistribution des votes conservés :")
        print(filtered["vote"].value_counts())

        print("\nDebugging sur les identifiants et pays :")
        print("Nombre de résolutions :", filtered["rcid"].nunique())
        print("Nombre de pays :", filtered["Country"].nunique())

        print("\nNombre de votes pour les résolutions :")
        print(filtered.groupby("rcid")["vote"].count().describe())

    # --- Country ISO Code Mapping ---
    name_to_iso = dict(
        zip(matcher["StateNme"].str.upper(), matcher["StateAbb"])
    )
    names_iso = {**name_to_iso, **HARDCODED_COUNTRY_MAP}

    filtered["country_code"] = filtered["Country"].map(names_iso)
    filtered["country_code"] = filtered["country_code"].fillna(filtered["Country"])

    return filtered


def transform_votes(votes: pd.DataFrame) -> pd.DataFrame:

    votes = votes.copy()

    matrix = votes[["country_code", "rcid", "session", "vote", "year"]]

    return matrix


def encoder_dictionnaries(vote_dataframe):

    sorted_countries = sorted(vote_dataframe["country_code"].unique())
    sessions = sorted(vote_dataframe["session"].unique())
    rcids = sorted(vote_dataframe["rcid"].unique())
    # years = sorted(vote_dataframe["year"].unique())

    country_to_idx = {country: idx for idx, country in enumerate(sorted_countries)}
    session_to_idx = {session: idx for idx, session in enumerate(sessions)}
    rcid_to_idx = {rcid: idx for idx, rcid in enumerate(rcids)}

    return country_to_idx, session_to_idx, rcid_to_idx


def structure_vote_table(vote_dataframe, country_dict, session_dict, rcid_dict):

    vote_dataframe = vote_dataframe.copy()
    vote_dataframe["country_code"] = vote_dataframe["country_code"].map(country_dict)
    vote_dataframe["session"] = vote_dataframe["session"].map(session_dict)
    vote_dataframe["rcid"] = vote_dataframe["rcid"].map(rcid_dict)

    # Faire un mapping pour correspondre à Ordered Probit. 
    # Dnas ce cadre le Yes serait plus élevé que le No en termes de valeurs
    vote_mapping = {
        1: 2,   # Yes -> 2
        2: 1,   # Abstain -> 1
        3: 0    # No -> 0
    }

    vote_dataframe["vote_ordinal"] = vote_dataframe["vote"].map(vote_mapping)

    vote_dataframe.rename(columns={"country_code":"country_idx", "session": "session_idx", 
    "rcid": "resolution_idx"}, inplace=True)

    return vote_dataframe


def build_session_years(df: pd.DataFrame) -> np.ndarray:
    return (df.groupby("session_idx")["year"].first().sort_index().values)
