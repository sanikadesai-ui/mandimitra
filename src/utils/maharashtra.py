"""
Maharashtra-specific constants and utilities for MANDIMITRA.
This module enforces strict Maharashtra-only constraints.
"""

from typing import Final

# =============================================================================
# HARD CONSTRAINT: MAHARASHTRA ONLY
# These constants MUST NOT be modified or overridden.
# The entire MANDIMITRA project exclusively serves Maharashtra farmers.
# =============================================================================

MAHARASHTRA_STATE_NAME: Final[str] = "Maharashtra"
MAHARASHTRA_STATE_CODE: Final[str] = "MH"

# All 36 districts of Maharashtra (official list)
MAHARASHTRA_DISTRICTS: Final[tuple] = (
    "Ahmednagar",
    "Akola",
    "Amravati",
    "Aurangabad",
    "Beed",
    "Bhandara",
    "Buldhana",
    "Chandrapur",
    "Dhule",
    "Gadchiroli",
    "Gondia",
    "Hingoli",
    "Jalgaon",
    "Jalna",
    "Kolhapur",
    "Latur",
    "Mumbai",
    "Mumbai City",
    "Mumbai Suburban",
    "Nagpur",
    "Nanded",
    "Nandurbar",
    "Nashik",
    "Osmanabad",
    "Palghar",
    "Parbhani",
    "Pune",
    "Raigad",
    "Ratnagiri",
    "Sangli",
    "Satara",
    "Sindhudurg",
    "Solapur",
    "Thane",
    "Wardha",
    "Washim",
    "Yavatmal",
)

# Maharashtra regional divisions
MAHARASHTRA_DIVISIONS: Final[dict] = {
    "Konkan": ["Mumbai City", "Mumbai Suburban", "Thane", "Palghar", "Raigad", "Ratnagiri", "Sindhudurg"],
    "Pune": ["Pune", "Satara", "Sangli", "Solapur", "Kolhapur"],
    "Nashik": ["Nashik", "Ahmednagar", "Dhule", "Nandurbar", "Jalgaon"],
    "Aurangabad": ["Aurangabad", "Jalna", "Beed", "Parbhani", "Hingoli", "Latur", "Osmanabad", "Nanded"],
    "Amravati": ["Amravati", "Akola", "Buldhana", "Washim", "Yavatmal"],
    "Nagpur": ["Nagpur", "Wardha", "Bhandara", "Gondia", "Chandrapur", "Gadchiroli"],
}


def is_maharashtra_state(state: str) -> bool:
    """
    Check if a state name matches Maharashtra (case-insensitive).
    
    Args:
        state: State name to check
        
    Returns:
        True if Maharashtra, False otherwise
    """
    if not state:
        return False
    return state.strip().lower() == MAHARASHTRA_STATE_NAME.lower()


def validate_maharashtra_only(state: str, strict: bool = True) -> bool:
    """
    Validate that a state is Maharashtra. Raises error in strict mode.
    
    Args:
        state: State name to validate
        strict: If True, raises ValueError for non-MH states
        
    Returns:
        True if Maharashtra
        
    Raises:
        ValueError: If strict=True and state is not Maharashtra
    """
    is_mh = is_maharashtra_state(state)
    
    if not is_mh and strict:
        raise ValueError(
            f"HARD CONSTRAINT VIOLATION: State '{state}' is not Maharashtra. "
            f"This project exclusively serves Maharashtra farmers. "
            f"Non-Maharashtra data is strictly prohibited."
        )
    
    return is_mh


def normalize_district_name(district: str) -> str:
    """
    Normalize Maharashtra district name for consistent matching.
    
    Args:
        district: District name (may have variations)
        
    Returns:
        Normalized district name
    """
    if not district:
        return ""
    
    # Common variations mapping
    variations = {
        "aurangabad": "Aurangabad",
        "chhatrapati sambhajinagar": "Aurangabad",
        "sambhajinagar": "Aurangabad",
        "osmanabad": "Osmanabad",
        "dharashiv": "Osmanabad",
        "mumbai city": "Mumbai City",
        "mumbai suburban": "Mumbai Suburban",
        "greater mumbai": "Mumbai",
    }
    
    normalized = district.strip().lower()
    return variations.get(normalized, district.strip().title())
