"""
Normalizes LLM-generated fact keys to canonical forms.
Prevents the same concept getting three different snake_case names
across episodes, which breaks find_existing key-match lookup.
"""

import re
from typing import Optional


# Canonical key map: any of these variations → single canonical key
# Add entries as you discover new collisions in your logs
KEY_SYNONYMS: dict[str, list[str]] = {
    "ui_theme": [
        "ui_theme",
        "ui_theme_preference",
        "interface_theme",
        "interface_theme_preference",
        "preferred_ui_theme",
        "theme_preference",
        "color_theme",
        "display_theme",
        "app_theme",
    ],
    "preferred_laptop": [
        "preferred_laptop",
        "laptop_preference",
        "computer_preference",
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
        "chat_tool",
    ],
    "version_control_system": [
        "version_control_system",
        "vcs",
        "git_platform",
        "code_repository",
        "source_control",
    ],
}

# Build reverse lookup: variant → canonical
_REVERSE: dict[str, str] = {}
for canonical, variants in KEY_SYNONYMS.items():
    for variant in variants:
        _REVERSE[variant.lower()] = canonical


def normalize_key(raw_key: str) -> str:
    """
    Normalize a raw LLM-generated key to its canonical form.
    
    Examples:
        "interface_theme_preference" → "ui_theme"
        "preferred_ui_theme"        → "ui_theme"
        "ui_theme_preference"       → "ui_theme"
        "manager_name"              → "manager_name"  (already canonical)
        "some_unknown_key"          → "some_unknown_key"  (no change)
    """
    cleaned = raw_key.lower().strip()
    canonical = _REVERSE.get(cleaned)
    if canonical:
        return canonical
    return cleaned


def keys_are_same_concept(key_a: str, key_b: str) -> bool:
    """Check if two keys refer to the same concept after normalization."""
    return normalize_key(key_a) == normalize_key(key_b)