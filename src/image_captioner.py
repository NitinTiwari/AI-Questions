"""
Handles image captioning and contextual synthesis for embedded images in Word documents.
Uses Groq Vision (e.g. llama-3.2-11b-vision-preview) when available, with persistent
local caching and contextual synthesis fallback so indexing never fails or blocks.
"""
import base64
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import settings

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = "captions_cache.json"


class ImageCaptioner:
    """Manages caption generation and caching for extracted images."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or settings.EXTRACTED_IMAGES_PATH)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / CACHE_FILE_NAME
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()
        self._groq_disabled = False

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read caption cache: {e}. Starting fresh.")
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save caption cache: {e}")

    def _encode_image_b64(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _try_groq_vision(self, image_path: Path, prompt: str) -> Optional[str]:
        if self._groq_disabled or not settings.GROQ_API_KEY:
            return None
        try:
            from groq import Groq
            client = Groq(api_key=settings.GROQ_API_KEY)
            b64_img = self._encode_image_b64(image_path)
            ext = image_path.suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

            response = client.chat.completions.create(
                model=settings.VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                    ]
                }],
                temperature=0.1,
                max_tokens=500,
                timeout=5.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.info(f"Groq vision unavailable ({e}). Falling back to contextual synthesis.")
            self._groq_disabled = True
            return None

    def generate_caption(
        self,
        image_path: Path | str,
        section: str,
        prev_text: str = "",
        next_text: str = "",
        source_doc: str = "",
        force_refresh: bool = False,
    ) -> str:
        """
        Return the caption for the given image, using cache when available,
        falling back to Groq Vision or contextual synthesis.
        """
        img_p = Path(image_path).resolve()
        cache_key = img_p.name

        if not force_refresh and cache_key in self.cache:
            return self.cache[cache_key]["caption"]

        prompt = (
            f"You are analyzing a technical diagram or illustration embedded in a document section titled '{section}'.\n"
            f"Preceding context: '{prev_text[:250]}'\n"
            f"Subsequent context: '{next_text[:250]}'\n\n"
            "Provide a concise, detailed technical description of this image. Include:\n"
            "1. The main architectural flow, components, or concept shown.\n"
            "2. Any text, formulas, or labels visible in the diagram.\n"
            "3. The primary takeaway or purpose of this figure."
        )

        caption = None
        method = "context_fallback"

        # 1. Try Groq Vision
        if not self._groq_disabled:
            caption = self._try_groq_vision(img_p, prompt)
            if caption:
                method = "groq_vision"

        # 2. Contextual synthesis fallback (deterministic & guaranteed)
        if not caption:
            context_parts = []
            if prev_text:
                context_parts.append(f"Preceded by: \"{prev_text}\"")
            if next_text:
                context_parts.append(f"Followed by: \"{next_text}\"")

            joined_context = " | ".join(context_parts) if context_parts else "Technical diagram."
            caption = (
                f"Technical diagram or visual illustration in section '{section}'. "
                f"Associated context: {joined_context}. "
                f"Referenced file: {img_p.name} from {source_doc}."
            )
            method = "context_fallback"

        # Save to cache
        self.cache[cache_key] = {
            "caption": caption,
            "section": section,
            "source": source_doc,
            "method": method,
            "path": str(img_p),
        }
        self._save_cache()
        return caption


# Global instance
captioner = ImageCaptioner()
