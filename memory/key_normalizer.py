"""
Normalizes LLM-generated fact keys to canonical forms.
Uses multi-level matching: exact → normalized → stem → LLM semantic.
"""

import re
from functools import lru_cache
from typing import Optional
from loguru import logger

# Lazy import to avoid circular dependencies
_llm_client = None
_llm_model = None


def _get_llm():
    """Lazy load LLM client to avoid import cycles."""
    global _llm_client, _llm_model
    if _llm_client is None:
        from memory.llm import llm_client, llm_model
        _llm_client = llm_client
        _llm_model = llm_model
    return _llm_client(), _llm_model()


# Manual overrides for ultra-common cases (fast path)
# Keep this list small - only top 10-20 most frequent keys
EXACT_SYNONYMS: dict[str, list[str]] = {
    "ui_theme": [
        "ui_theme",
        "ui_theme_preference",
        "interface_theme",
        "interface_theme_preference",
        "theme_preference",
        "display_theme",
    ],
    "preferred_laptop": [
        "preferred_laptop",
        "laptop_preference",
        "device_preference",
        "preferred_device",
    ],
    "manager_name": [
        "manager_name",
        "direct_manager",
        "reporting_manager",
        "manager",
    ],
    "it_communication_tool": [
        "it_communication_tool",
        "communication_tool",
        "team_communication_tool",
        "messaging_tool",
    ],
    "version_control_system": [
        "version_control_system",
        "vcs",
        "git_platform",
        "source_control",
    ],
}

# Build reverse lookup: variant → canonical
_REVERSE: dict[str, str] = {}
for canonical, variants in EXACT_SYNONYMS.items():
    for variant in variants:
        _REVERSE[variant.lower()] = canonical


def normalize_key(raw_key: str) -> str:
    """
    Fast-path normalization using exact synonym matches.
    
    Examples:
        "interface_theme_preference" → "ui_theme"
        "laptop_preference"         → "preferred_laptop"
        "unknown_key"               → "unknown_key" (no change)
    """
    cleaned = raw_key.lower().strip()
    return _REVERSE.get(cleaned, cleaned)


@lru_cache(maxsize=1000)
def keys_are_same_concept(key_a: str, key_b: str) -> bool:
    """
    Multi-level semantic matching to determine if two keys refer to same concept.
    
    Levels (in order, stops at first match):
    1. Exact match - "ui_theme" == "ui_theme"
    2. Normalized match - "interface_theme" → "ui_theme" (via EXACT_SYNONYMS)
    3. Stem overlap - both contain "laptop" → likely same
    4. LLM semantic match - ask GPT if they're equivalent (cached)
    
    Args:
        key_a: First fact key (e.g., "ui_theme")
        key_b: Second fact key (e.g., "interface_theme_preference")
    
    Returns:
        True if keys refer to same concept, False otherwise
    """
    if not key_a or not key_b:
        return False
    
    # Level 1: Exact match
    if key_a == key_b:
        return True
    
    # Level 2: Normalized match (via manual synonym map)
    norm_a = normalize_key(key_a)
    norm_b = normalize_key(key_b)
    
    if norm_a == norm_b:
        logger.debug(
            f"[KeyNormalizer] Normalized match: '{key_a}' == '{key_b}' "
            f"→ '{norm_a}'"
        )
        return True
    
    # Level 3: Stem overlap heuristic
    # Extract words of 4+ chars, take first 4 letters as stem
    stems_a = {w[:4] for w in re.findall(r'[a-z]{4,}', norm_a)}
    stems_b = {w[:4] for w in re.findall(r'[a-z]{4,}', norm_b)}
    
    if stems_a and stems_b:
        overlap = stems_a & stems_b
        if overlap:
            union = stems_a | stems_b
            overlap_ratio = len(overlap) / len(union)
            
            # If >60% stem overlap, consider them same concept
            # e.g., "laptop_preference" vs "preferred_laptop" both have "lapt", "pref"
            if overlap_ratio > 0.6:
                logger.debug(
                    f"[KeyNormalizer] Stem match ({overlap_ratio:.1%}): "
                    f"'{key_a}' ≈ '{key_b}' (stems: {overlap})"
                )
                return True
    
    # Level 4: LLM semantic match (expensive, but cached)
    return _llm_keys_match(norm_a, norm_b)


@lru_cache(maxsize=500)
def _llm_keys_match(key_a: str, key_b: str) -> bool:
    """
    Use LLM to determine if two keys are semantically equivalent.
    Results are cached to avoid repeated API calls.
    
    Only called when exact/normalized/stem matching fails.
    
    Args:
        key_a: First normalized key
        key_b: Second normalized key
    
    Returns:
        True if LLM confirms they're the same concept
    """
    
    prompt = f"""Are these two fact keys referring to the SAME underlying concept?

Key A: {key_a}
Key B: {key_b}

Examples of SAME concept:
- "ui_theme" vs "interface_theme_preference" → YES (both about UI color scheme)
- "manager_name" vs "direct_manager" → YES (both about who you report to)
- "laptop_preference" vs "preferred_laptop" → YES (same thing, different wording)

Examples of DIFFERENT concepts:
- "ui_theme" vs "manager_name" → NO (completely different facts)
- "laptop_model" vs "laptop_manufacturer" → NO (model vs brand)
- "preferred_laptop" vs "assigned_laptop" → NO (preference vs actual device)

Answer with ONLY one word: YES or NO"""

    try:
        client, model = _get_llm()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a semantic equivalence checker. Answer only YES or NO."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=5
        )
        
        answer = resp.choices[0].message.content.strip().upper()
        result = "YES" in answer
        
        logger.debug(
            f"[KeyNormalizer] LLM semantic match: '{key_a}' vs '{key_b}' → {result}"
        )
        return result
        
    except Exception as exc:
        logger.warning(
            f"[KeyNormalizer] LLM call failed for '{key_a}' vs '{key_b}': {exc}. "
            f"Defaulting to NO MATCH (conservative)"
        )
        # Conservative fallback - don't assume they match
        return False


def add_synonym_rule(canonical: str, *variants: str) -> None:
    """
    Dynamically add a new synonym rule at runtime.
    Useful for learning from corrections.
    
    Example:
        add_synonym_rule("ui_theme", "color_scheme", "app_theme")
    """
    if canonical not in EXACT_SYNONYMS:
        EXACT_SYNONYMS[canonical] = [canonical]
    
    for variant in variants:
        variant_lower = variant.lower()
        if variant_lower not in EXACT_SYNONYMS[canonical]:
            EXACT_SYNONYMS[canonical].append(variant_lower)
        _REVERSE[variant_lower] = canonical
    
    # Clear cache so new rules take effect
    keys_are_same_concept.cache_clear()
    
    logger.info(
        f"[KeyNormalizer] Added synonym rule: {variants} → '{canonical}'"
    )


def get_canonical_key(raw_key: str) -> str:
    """
    Get the canonical form of a key, creating new canonical if unknown.
    This is used when storing NEW facts - we normalize to prevent drift.
    
    Returns the existing canonical if found in synonyms, otherwise returns
    the cleaned input as the new canonical form.
    """
    return normalize_key(raw_key)