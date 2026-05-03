PRETRAIN_TICKERS = [
    "OTEX",
    "PATH",
    "PUBM",
    "GOOGL",
    "ADP",
    "CBRE",
    "BLK",
    "FRT",
    "PINS",
    "MKTX",
]

EVAL_TICKERS = [
    "INTA",
    "FFIV",
    "MLM",
    "PARR",
    "SEIC",
]

# Curated metadata used for sector-aware retrieval scoring.
# NAICS codes are broad/company-level approximations adequate for metadata conditioning.
TICKER_METADATA = {
    "OTEX": {"sector": "Technology", "industry": "Software", "naics": "511210"},
    "PATH": {"sector": "Technology", "industry": "Software", "naics": "511210"},
    "PUBM": {"sector": "Technology", "industry": "Software", "naics": "511210"},
    "GOOGL": {"sector": "Communication Services", "industry": "Internet Content", "naics": "519290"},
    "ADP": {"sector": "Technology", "industry": "Data Processing", "naics": "518210"},
    "CBRE": {"sector": "Real Estate", "industry": "Real Estate Services", "naics": "531210"},
    "BLK": {"sector": "Financials", "industry": "Asset Management", "naics": "523920"},
    "FRT": {"sector": "Real Estate", "industry": "Retail REIT", "naics": "531120"},
    "PINS": {"sector": "Communication Services", "industry": "Internet Content", "naics": "519290"},
    "MKTX": {"sector": "Financials", "industry": "Capital Markets", "naics": "523140"},
    "INTA": {"sector": "Technology", "industry": "Software", "naics": "511210"},
    "FFIV": {"sector": "Technology", "industry": "Software", "naics": "511210"},
    "MLM": {"sector": "Materials", "industry": "Construction Materials", "naics": "327310"},
    "PARR": {"sector": "Energy", "industry": "Refining & Marketing", "naics": "324110"},
    "SEIC": {"sector": "Financials", "industry": "Asset Management", "naics": "523920"},
}


def metadata_for_ticker(ticker: str) -> dict:
    meta = TICKER_METADATA.get(ticker, {})
    naics = str(meta.get("naics", "unknown"))
    return {
        "sector": meta.get("sector", "unknown"),
        "industry": meta.get("industry", "unknown"),
        "naics": naics,
        "naics2": naics[:2] if naics != "unknown" else "unknown",
    }

