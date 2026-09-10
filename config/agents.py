"""Hardcoded Agent Configurations for HR and Onboarding domains."""

AGENT_CONFIGS = {
    "hr_assistant": {
        "agent_id": "hr_assistant_001",
        "name": "HR Assistant",
        "description": "Assists employees with corporate HR policies, time off, and benefits queries.",
        "role": (
            "You are a knowledgeable and empathetic Corporate HR Assistant. "
            "You guide staff through policies, procedures, and internal benefits with accuracy."
        ),
        "knowledge_base": [
            {
                "title": "General HR Policies",
                "category": "policy",
                "content": (
                    "# Corporate HR Policy Guidelines\n\n"
                    "## Working Hours & Core Window\n"
                    "Standard work hours are 40 hours per week. Core collaboration hours are 10:00 AM to 3:00 PM local time.\n\n"
                    "## Hybrid Work Policy\n"
                    "Employees are permitted two remote days per week. Three days must be spent in their assigned regional office.\n\n"
                    "## Performance Appraisals\n"
                    "Performance appraisals occur bi-annually in June and December."
                )
            },
            {
                "title": "Paid Time Off (PTO) Framework",
                "category": "pto",
                "content": (
                    "# Paid Time Off Guidelines\n\n"
                    "## Accrual Rates\n"
                    "Full-time employees accrue 1.25 PTO days per month (15 days annually).\n\n"
                    "## Carryover Rules\n"
                    "A maximum of 5 unused PTO days can roll over into the subsequent fiscal calendar year. Surplus balance lapses on January 31.\n\n"
                    "## Extended Leave Notice\n"
                    "Any PTO request exceeding 5 consecutive business days requires at least two weeks of prior written notice."
                )
            }
        ],
        "procedural_skills": [
            {
                "title": "Submit PTO Request",
                "description": "Standard operating procedure to log vacation or personal time off.",
                "scope": "shared:organization",
                "status": "approved",
                "content": (
                    "# Standard Procedure: Submit PTO Request\n\n"
                    "1. Access the enterprise portal at `https://workday.company.internal`.\n"
                    "2. Authenticate using your Single Sign-On (SSO) credentials and 2FA.\n"
                    "3. Select the 'Time Off' icon from the main dashboard.\n"
                    "4. Choose 'Request Absence' from the left navigation rail.\n"
                    "5. Select the calendar date range and tag the category (Vacation/Sick/Personal).\n"
                    "6. Enter your line manager's corporate email in the routing field.\n"
                    "7. Click 'Submit' and confirm verification dialogue.\n\n"
                    "**Crucial Constraint**: If your absence span exceeds 5 consecutive business days, "
                    "two weeks advance submission is strictly enforced by corporate policy."
                )
            }
        ],
        "semantic_facts": [
            {
                "content": "PTO requests over 5 days require 2 weeks advance notice.",
                "scope": "shared:organization",
                "confidence": 1.0
            },
            {
                "content": "Health insurance open enrollment takes place annually from November 1 through November 30.",
                "scope": "shared:organization",
                "confidence": 1.0
            },
            {
                "content": "The standard corporate 401(k) matching tier is 100% matching up to 4% of base pay.",
                "scope": "shared:organization",
                "confidence": 1.0
            }
        ],
        "memory_config": {
            "consolidation_triggers": {
                "session_end": True,
                "user_feedback": True,
                "event_count_threshold": 20
            },
            "promotion_settings": {
                "semantic_auto_approve_threshold": 0.85,
                "procedural_requires_hitl": True
            }
        }
    },
    "onboarding_assistant": {
        "agent_id": "onboarding_assistant_001",
        "name": "Onboarding Assistant",
        "description": "Guides new hires through setup, equipment provisioning, and security clearance.",
        "role": (
            "You are an energetic Onboarding Coordinator. "
            "You help new team members navigate their first 30 days and configure accounts smoothly."
        ),
        "knowledge_base": [
            {
                "title": "New Hire First Week Checklist",
                "category": "checklist",
                "content": (
                    "# First Week Checklist for New Team Members\n\n"
                    "## Day 1\n"
                    "- Complete I-9 verification with Identity Services.\n"
                    "- Obtain hardware credential pass from IT Helpdesk.\n"
                    "- Join #welcome-team Slack room.\n\n"
                    "## Day 2-3\n"
                    "- Complete Mandatory Compliance Training modules 1 through 4.\n"
                    "- Meet with your department mentor."
                )
            }
        ],
        "procedural_skills": [
            {
                "title": "Provision Developer Environment",
                "description": "Step-by-step instructions to configure engineering laptops.",
                "scope": "shared:organization",
                "status": "approved",
                "content": (
                    "# Procedure: Provision Developer Environment\n\n"
                    "1. Connect machine to `Corp-Secure-WiFi` using device profile certificates.\n"
                    "2. Run `/bin/bash -c \"$(curl -fsSL https://setup.internal.dev/bootstrap.sh)\"`.\n"
                    "3. Open Okta Verify app and register hardware YubiKey.\n"
                    "4. Clone repository via SSH key registered in internal GitHub Enterprise."
                )
            }
        ],
        "semantic_facts": [
            {
                "content": "New employees receive their primary hardware dispatch 3 business days before their start date.",
                "scope": "shared:organization",
                "confidence": 1.0
            }
        ],
        "memory_config": {
            "consolidation_triggers": {
                "session_end": True,
                "event_count_threshold": 15
            },
            "promotion_settings": {
                "semantic_auto_approve_threshold": 0.85,
                "procedural_requires_hitl": True
            }
        }
    }
}
