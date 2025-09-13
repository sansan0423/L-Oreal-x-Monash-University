# superficial.py
# Functions to detect superficial (low-value) vs substantive comments

import re
import emoji

# --- Generic filler words / phrases (expand if needed) ---
GENERIC_WORDS = {"nice", "good", "great", "cool", "wow", "love it", "🔥", "😍", "👍"}

def is_superficial(raw_text: str) -> bool:
    """
    Returns True if the comment is superficial (spam/emoji/too short).
    Otherwise False (substantive).
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)
    text = raw_text.strip().lower()

    # 1. Empty
    if len(text) == 0:
        return True

    # 2. Emoji-only (remove all emojis, if nothing left → superficial)
    text_no_emoji = emoji.replace_emoji(text, replace="")
    if len(text_no_emoji.strip()) == 0:
        return True

    # 3. Very short (≤ 2 words)
    words = re.findall(r"\w+", text_no_emoji)
    if len(words) <= 2:
        return True

    # 4. Generic filler (single common word/phrase)
    if text in GENERIC_WORDS:
        return True

    return False
