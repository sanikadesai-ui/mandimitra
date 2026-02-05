#!/usr/bin/env python3
"""
MANDIMITRA - District Name Normalization

Maps raw district names from various sources to canonical Maharashtra district names.
Uses configs/maharashtra_locations.csv as the authoritative source of 36 districts.

Key Features:
- Deterministic mapping (no fuzzy matching in production mode)
- Handles common variants (Mumbai vs Mumbai City, Sholapur vs Solapur, etc.)
- Reports unmapped districts for manual resolution
- Creates dim_districts.csv dimension table

Usage:
    from src.utils.district_normalize import DistrictNormalizer
    
    normalizer = DistrictNormalizer()
    canonical = normalizer.normalize("Sholapur")  # Returns "Solapur"
    
    # Or normalize a DataFrame column
    df["district_canonical"] = df["district_raw"].apply(normalizer.normalize)
"""

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# CANONICAL DISTRICT LIST (36 Maharashtra Districts)
# =============================================================================

# These are the official 36 districts from configs/maharashtra_locations.csv
CANONICAL_DISTRICTS: List[str] = [
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
]

# =============================================================================
# DISTRICT ALIAS MAPPING (Raw -> Canonical)
# =============================================================================

# Deterministic mapping from known variants to canonical names
# Key: lowercase variant, Value: canonical name
DISTRICT_ALIAS_MAP: Dict[str, str] = {
    # Exact matches (lowercase)
    "ahmednagar": "Ahmednagar",
    "akola": "Akola",
    "amravati": "Amravati",
    "aurangabad": "Aurangabad",
    "beed": "Beed",
    "bhandara": "Bhandara",
    "buldhana": "Buldhana",
    "chandrapur": "Chandrapur",
    "dhule": "Dhule",
    "gadchiroli": "Gadchiroli",
    "gondia": "Gondia",
    "hingoli": "Hingoli",
    "jalgaon": "Jalgaon",
    "jalna": "Jalna",
    "kolhapur": "Kolhapur",
    "latur": "Latur",
    "mumbai city": "Mumbai City",
    "mumbai suburban": "Mumbai Suburban",
    "nagpur": "Nagpur",
    "nanded": "Nanded",
    "nandurbar": "Nandurbar",
    "nashik": "Nashik",
    "osmanabad": "Osmanabad",
    "palghar": "Palghar",
    "parbhani": "Parbhani",
    "pune": "Pune",
    "raigad": "Raigad",
    "ratnagiri": "Ratnagiri",
    "sangli": "Sangli",
    "satara": "Satara",
    "sindhudurg": "Sindhudurg",
    "solapur": "Solapur",
    "thane": "Thane",
    "wardha": "Wardha",
    "washim": "Washim",
    "yavatmal": "Yavatmal",
    
    # --- COMMON VARIANTS / ALIASES ---
    
    # Amravati variants
    "amarawati": "Amravati",
    "amrawati": "Amravati",
    
    # Aurangabad / Chhatrapati Sambhajinagar (renamed in 2022)
    "chattrapati sambhajinagar": "Aurangabad",
    "chhatrapati sambhajinagar": "Aurangabad",
    "sambhajinagar": "Aurangabad",
    "chh. sambhajinagar": "Aurangabad",
    
    # Osmanabad / Dharashiv (renamed in 2022)
    "dharashiv": "Osmanabad",
    "dharashiv (usmanabad)": "Osmanabad",
    "dharashiv(usmanabad)": "Osmanabad",
    "usmanabad": "Osmanabad",
    
    # Gondia variants
    "gondiya": "Gondia",
    "gondya": "Gondia",
    
    # Jalna variants
    "jalana": "Jalna",
    
    # Solapur variants
    "sholapur": "Solapur",
    "sholapur.": "Solapur",
    
    # Washim variants
    "vashim": "Washim",
    
    # Mumbai variants (most mandi data just says "Mumbai")
    "mumbai": "Mumbai City",  # Default Mumbai -> Mumbai City
    "greater mumbai": "Mumbai Suburban",
    "mumbai (suburban)": "Mumbai Suburban",
    
    # Beed variants
    "bid": "Beed",
    
    # Nashik variants
    "nasik": "Nashik",
    
    # Raigad variants
    "raigarh": "Raigad",
    "alibag": "Raigad",
    
    # Sindhudurg variants
    "sindhdurg": "Sindhudurg",
    "sindhudurga": "Sindhudurg",
    
    # Nanded variants
    "nanded-waghala": "Nanded",
    
    # Parbhani variants
    "parbani": "Parbhani",
    
    # Buldhana variants
    "buldana": "Buldhana",
    
    # Chandrapur variants
    "chandrapura": "Chandrapur",
    
    # Gadchiroli variants
    "gadchiroli.": "Gadchiroli",
    
    # Hingoli variants
    "hingholi": "Hingoli",
    
    # Yavatmal variants
    "yeotmal": "Yavatmal",
    "yavtmal": "Yavatmal",
    
    # Kolhapur variants
    "kolhapura": "Kolhapur",
    
    # Unknown/Special - Map to nearest or report
    "murum": "Osmanabad",  # Murum is a taluka in Osmanabad district
}


class DistrictNormalizer:
    """
    Normalizes district names to canonical Maharashtra district names.
    
    Attributes:
        canonical_districts: Set of 36 official district names
        alias_map: Dict mapping lowercase variants to canonical names
        unmapped: Set of district names that couldn't be mapped
        strict: If True, raise error on unmapped districts
    """
    
    def __init__(
        self,
        locations_csv: Optional[Path] = None,
        strict: bool = False,
        allow_unmapped: bool = False,
    ):
        """
        Initialize the normalizer.
        
        Args:
            locations_csv: Path to maharashtra_locations.csv (uses default if None)
            strict: If True, raise ValueError on unmapped districts
            allow_unmapped: If True, return original name for unmapped (logs warning)
        """
        self.strict = strict
        self.allow_unmapped = allow_unmapped
        self.unmapped: Set[str] = set()
        self._mapping_stats: Dict[str, int] = {}
        
        # Load canonical districts from CSV if provided
        if locations_csv is None:
            locations_csv = PROJECT_ROOT / "configs" / "maharashtra_locations.csv"
        
        self.canonical_districts = set(CANONICAL_DISTRICTS)
        self.alias_map = DISTRICT_ALIAS_MAP.copy()
        
        # Load from CSV to ensure sync
        if locations_csv.exists():
            self._load_from_csv(locations_csv)
        
        # Build reverse lookup for canonical names (lowercase -> canonical)
        for canonical in self.canonical_districts:
            self.alias_map[canonical.lower()] = canonical
    
    def _load_from_csv(self, csv_path: Path) -> None:
        """Load canonical districts from maharashtra_locations.csv."""
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                # Skip comment lines
                lines = [line for line in f if not line.startswith("#")]
            
            reader = csv.DictReader(lines)
            csv_districts = set()
            for row in reader:
                district = row.get("district", "").strip()
                if district:
                    csv_districts.add(district)
                    # Add lowercase mapping
                    self.alias_map[district.lower()] = district
            
            if csv_districts:
                self.canonical_districts = csv_districts
                logger.debug(f"Loaded {len(csv_districts)} districts from {csv_path}")
        except Exception as e:
            logger.warning(f"Could not load districts from {csv_path}: {e}")
    
    def normalize(self, district_raw: str) -> Optional[str]:
        """
        Normalize a district name to its canonical form.
        
        Args:
            district_raw: Raw district name from data source
            
        Returns:
            Canonical district name, or None if unmapped and not allow_unmapped
            
        Raises:
            ValueError: If strict=True and district cannot be mapped
        """
        if not district_raw or not isinstance(district_raw, str):
            return None
        
        # Clean and lowercase for lookup
        cleaned = district_raw.strip().lower()
        
        # Remove common suffixes/prefixes
        cleaned = cleaned.replace("district", "").strip()
        cleaned = cleaned.replace("dist.", "").strip()
        cleaned = cleaned.replace("dist", "").strip()
        
        # Try exact match in alias map
        if cleaned in self.alias_map:
            canonical = self.alias_map[cleaned]
            self._mapping_stats[canonical] = self._mapping_stats.get(canonical, 0) + 1
            return canonical
        
        # Try without parentheses content
        if "(" in cleaned:
            base = cleaned.split("(")[0].strip()
            if base in self.alias_map:
                canonical = self.alias_map[base]
                self._mapping_stats[canonical] = self._mapping_stats.get(canonical, 0) + 1
                return canonical
        
        # Unmapped district
        self.unmapped.add(district_raw)
        
        if self.strict:
            raise ValueError(f"Unmapped district: '{district_raw}' - add to DISTRICT_ALIAS_MAP")
        
        if self.allow_unmapped:
            logger.warning(f"Unmapped district (returning as-is): '{district_raw}'")
            return district_raw
        
        logger.warning(f"Unmapped district (returning None): '{district_raw}'")
        return None
    
    def normalize_batch(self, districts: List[str]) -> List[Optional[str]]:
        """Normalize a list of district names."""
        return [self.normalize(d) for d in districts]
    
    def get_unmapped_report(self) -> str:
        """Generate a Markdown report of unmapped districts."""
        if not self.unmapped:
            return "# Unmapped Districts Report\n\n✅ All districts mapped successfully.\n"
        
        lines = [
            "# Unmapped Districts Report",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Total Unmapped:** {len(self.unmapped)}",
            "",
            "## Unmapped District Names",
            "",
            "| Raw District Name | Suggested Canonical |",
            "|-------------------|---------------------|",
        ]
        
        for raw in sorted(self.unmapped):
            # Try to suggest based on partial match
            suggestion = self._suggest_canonical(raw)
            lines.append(f"| {raw} | {suggestion or '???'} |")
        
        lines.extend([
            "",
            "## Action Required",
            "",
            "Add these mappings to `src/utils/district_normalize.py` in `DISTRICT_ALIAS_MAP`:",
            "",
            "```python",
        ])
        
        for raw in sorted(self.unmapped):
            suggestion = self._suggest_canonical(raw) or "???"
            lines.append(f'    "{raw.lower()}": "{suggestion}",')
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def _suggest_canonical(self, raw: str) -> Optional[str]:
        """Suggest a canonical district based on partial matching."""
        raw_lower = raw.lower()
        
        # Try prefix match
        for canonical in self.canonical_districts:
            if canonical.lower().startswith(raw_lower[:3]):
                return canonical
            if raw_lower.startswith(canonical.lower()[:3]):
                return canonical
        
        return None
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics on how many times each canonical district was mapped."""
        return self._mapping_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset mapping statistics and unmapped set."""
        self._mapping_stats.clear()
        self.unmapped.clear()


def build_dim_districts(
    output_path: Optional[Path] = None,
    locations_csv: Optional[Path] = None,
) -> Path:
    """
    Build the dim_districts.csv dimension table.
    
    Args:
        output_path: Where to save the dimension table
        locations_csv: Path to maharashtra_locations.csv
        
    Returns:
        Path to the created dimension table
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "processed" / "dim_districts.csv"
    
    if locations_csv is None:
        locations_csv = PROJECT_ROOT / "configs" / "maharashtra_locations.csv"
    
    # Build reverse alias map (canonical -> list of aliases)
    aliases_map: Dict[str, List[str]] = {d: [] for d in CANONICAL_DISTRICTS}
    for alias, canonical in DISTRICT_ALIAS_MAP.items():
        if canonical in aliases_map and alias != canonical.lower():
            aliases_map[canonical].append(alias)
    
    # Load coordinates from locations CSV
    coords: Dict[str, Tuple[float, float]] = {}
    if locations_csv.exists():
        with open(locations_csv, "r", encoding="utf-8") as f:
            lines = [line for line in f if not line.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            district = row.get("district", "").strip()
            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
            coords[district] = (lat, lon)
    
    # Write dimension table
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_district", "aliases", "latitude", "longitude"])
        
        for district in sorted(CANONICAL_DISTRICTS):
            alias_list = "|".join(sorted(set(aliases_map.get(district, []))))
            lat, lon = coords.get(district, (0.0, 0.0))
            writer.writerow([district, alias_list, lat, lon])
    
    logger.info(f"Created dim_districts.csv with {len(CANONICAL_DISTRICTS)} districts: {output_path}")
    return output_path


def save_unmapped_report(
    normalizer: DistrictNormalizer,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Save unmapped districts report to logs directory.
    
    Args:
        normalizer: DistrictNormalizer instance with unmapped data
        output_dir: Directory to save report (default: logs/)
        
    Returns:
        Path to report file, or None if no unmapped districts
    """
    if not normalizer.unmapped:
        return None
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"unmapped_districts_{timestamp}.md"
    
    report_content = normalizer.get_unmapped_report()
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.warning(f"Unmapped districts report saved: {report_path}")
    return report_path


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# Global normalizer instance (lazy initialization)
_global_normalizer: Optional[DistrictNormalizer] = None


def get_normalizer(strict: bool = False) -> DistrictNormalizer:
    """Get the global district normalizer instance."""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = DistrictNormalizer(strict=strict)
    return _global_normalizer


def normalize_district(district_raw: str) -> Optional[str]:
    """Normalize a single district name using the global normalizer."""
    return get_normalizer().normalize(district_raw)


def get_canonical_districts() -> List[str]:
    """Get the list of 36 canonical Maharashtra district names."""
    return CANONICAL_DISTRICTS.copy()


if __name__ == "__main__":
    # Test the normalizer
    print("Testing District Normalizer")
    print("=" * 50)
    
    normalizer = DistrictNormalizer()
    
    test_cases = [
        "Sholapur",
        "sholapur",
        "Gondiya",
        "Amarawati",
        "Chattrapati Sambhajinagar",
        "Dharashiv (Usmanabad)",
        "Mumbai",
        "Vashim",
        "Jalana",
        "Murum",
        "Unknown District",
    ]
    
    for raw in test_cases:
        canonical = normalizer.normalize(raw)
        print(f"  {raw:35} -> {canonical}")
    
    print()
    print("Unmapped districts:", normalizer.unmapped)
    
    # Build dimension table
    print()
    print("Building dim_districts.csv...")
    dim_path = build_dim_districts()
    print(f"Created: {dim_path}")
