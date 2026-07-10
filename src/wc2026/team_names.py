"""Canonical team-name normalization and the 48 qualified WC 2026 teams."""

from __future__ import annotations

# 48 qualified teams in canonical names.
QUALIFIED_TEAMS_2026: frozenset[str] = frozenset(
    [
        "Mexico",
        "South Africa",
        "South Korea",
        "Czech Republic",
        "Canada",
        "Bosnia and Herzegovina",
        "Qatar",
        "Switzerland",
        "Brazil",
        "Morocco",
        "Haiti",
        "Scotland",
        "United States",
        "Paraguay",
        "Australia",
        "Turkey",
        "Germany",
        "Curacao",
        "Ivory Coast",
        "Ecuador",
        "Netherlands",
        "Japan",
        "Sweden",
        "Tunisia",
        "Belgium",
        "Egypt",
        "Iran",
        "New Zealand",
        "Spain",
        "Cabo Verde",
        "Saudi Arabia",
        "Uruguay",
        "France",
        "Senegal",
        "Iraq",
        "Norway",
        "Argentina",
        "Algeria",
        "Austria",
        "Jordan",
        "Portugal",
        "DR Congo",
        "Uzbekistan",
        "Colombia",
        "England",
        "Croatia",
        "Ghana",
        "Panama",
    ]
)

# API-Football / common aliases -> international_results canonical name
TEAM_ALIASES: dict[str, str] = {
    # Task-specific fixture mappings
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "Curaçao": "Curacao",
    "USA": "United States",
    "Cape Verde": "Cabo Verde",
    "Korea Republic": "South Korea",
    "United States": "United States",
    "DPR Korea": "North Korea",
    "Republic of Ireland": "Ireland",

    # Wider alias support
    "Republic of Korea": "South Korea",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cabo Verde",
    "Cabo Verde": "Cabo Verde",
    "Congo": "Congo",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",

    # Keep common but non-qualified normalization (already used elsewhere)
    "Cape Verde Islands": "Cabo Verde",
    "Chinese Taipei": "Chinese Taipei",
    "Hong Kong, China": "Hong Kong",
    "Turkey": "Turkey",
    "DR Congo": "DR Congo",

    # If you see these later in fixtures/results, keep as-is unless you want to map them.
    "Russia": "Russia",

    # If international_results uses these exact names, mapping to themselves is harmless.
    "Curacao": "Curacao",
}


def canonical_team(name: str) -> str:
    """Return canonical team name used in international_results."""
    if not name:
        return name
    stripped = name.strip()
    return TEAM_ALIASES.get(stripped, stripped)


def is_qualified_team(name: str) -> bool:
    """True if the given (possibly aliased) team is one of the 48 qualified teams."""
    return canonical_team(name) in QUALIFIED_TEAMS_2026

