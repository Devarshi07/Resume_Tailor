import re

# Extract and replace bullets in a given section

def extract_section(resume: str, section_name: str) -> str:
    """
    Find the text under the given section heading (e.g., "Work Experience").
    Captures everything until the next all-caps-ish heading or end of document.
    """
    pattern = rf"(?mi)(?P<header>{re.escape(section_name)}\s*\n)(?P<body>.*?)(?=\n[A-Z][A-Z ]+\n|\Z)"
    match = re.search(pattern, resume)
    return match.group("body") if match else ""


def replace_section(resume: str, section_name: str, new_body: str) -> str:
    """
    Replace the entire body under section_name (including old bullets) with new_body.
    Keeps the section header intact.
    """
    pattern = rf"(?mi)(?P<header>{re.escape(section_name)}\s*\n)(?P<body>.*?)(?=\n[A-Z][A-Z ]+\n|\Z)"
    def _repl(m):
        return m.group("header") + new_body
    return re.sub(pattern, _repl, resume)


def extract_bullets(section_text: str) -> list[str]:
    """
    From a section’s raw text, collect all lines that look like bullet points.
    We accept lines starting with: - “- ” or “* ” or “• ” (possibly preceded by tabs/spaces).
    """
    bullets = re.findall(r"^[\t ]*(?:[-\*\u2022])\s+.*", section_text, flags=re.MULTILINE)
    return bullets


def replace_bullets(section_text: str, new_bullets: list[str]) -> str:
    """
    Replace the old bullet lines in section_text one‐for‐one with new_bullets.
    Non-bullet lines remain unchanged.
    """
    lines = section_text.splitlines()
    out_lines = []
    bi = 0
    for line in lines:
        if re.match(r"^[\t ]*(?:[-\*\u2022])\s+.*", line):
            if bi < len(new_bullets):
                out_lines.append(new_bullets[bi])
                bi += 1
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
    while bi < len(new_bullets):
        out_lines.append(new_bullets[bi])
        bi += 1
    return "\n".join(out_lines)
