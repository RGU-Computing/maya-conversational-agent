import re


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax for TTS-friendly plain text."""
    if not text:
        return ""

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    # Inline code: `code` -> code
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Images: ![alt](url) -> ""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Links: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Bold/italic: **text**, __text__, *text*, _text_ -> text
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)

    # Headings: "# Title" -> "Title"
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Blockquotes: "> text" -> "text"
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)

    # Unordered list markers: "- text", "* text", "+ text" -> "text"
    text = re.sub(r"^\s*[-*+]\s+(?=\S)", "", text, flags=re.MULTILINE)

    # Ordered list markers: "1. text" -> "text"
    text = re.sub(r"^\s*\d+\.\s+(?=\S)", "", text, flags=re.MULTILINE)

    # Collapse excess whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def strip_citations_and_references(text: str) -> str:
    """Remove inline numeric citations like [1][2] and the References section,
    but keep any Safety Disclaimer / Disclaimer blocks.

    Designed for medical-style answers with 'References:' and inline [n] cites.
    """
    if not text:
        return text

    cleaned = text

    # 1) Remove inline numeric citations like [1], [1, 2], [2-4], [1, 3-5]
    inline_cite_pattern = re.compile(
        r'\[(?:\s*\d+\s*(?:[-–]\s*\d+)?\s*(?:,\s*\d+\s*(?:[-–]\s*\d+)?\s*)*)\]'
    )
    cleaned = inline_cite_pattern.sub('', cleaned)

    # 2) Remove the References section but keep any disclaimer after it
    reference_heading = re.search(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?references[ \t]*(?::|$)",
        cleaned,
    )
    if reference_heading is not None:
        ref_idx = reference_heading.start()
        lower = cleaned.lower()
        # Look for a disclaimer AFTER the References section
        tail_lower = lower[ref_idx:]

        safety_rel = tail_lower.find("safety disclaimer")
        # generic 'disclaimer' as a fallback (but after 'References:')
        generic_rel = tail_lower.find("disclaimer") if safety_rel == -1 else -1

        if safety_rel != -1:
            # Keep everything from the disclaimer onwards
            split_idx = ref_idx + safety_rel
        elif generic_rel != -1:
            split_idx = ref_idx + generic_rel
        else:
            # No disclaimer found → just cut everything from References: to end
            cleaned = cleaned[:ref_idx].rstrip()
            split_idx = None

        if split_idx is not None:
            before = cleaned[:ref_idx].rstrip()
            after = cleaned[split_idx:].lstrip()
            cleaned = before + "\n\n" + after

    # 3) Clean up spaces and excessive blank lines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)          # collapse multiple spaces
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)  # 3+ blank lines → 2

    return cleaned.strip()
