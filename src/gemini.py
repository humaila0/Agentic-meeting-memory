
"""Gemini wrapper with rate limiting, retries and quota protection."""

import json
import re
import time

from google import genai
from google.genai import types
from .config import GEMINI_API_KEY, GEMINI_MODEL


# Keep requests spaced out. Adjust this to your actual model quota.
MIN_SECONDS_BETWEEN_REQUESTS = 5.0
MAX_RETRIES = 3

_last_request_at = 0.0


class Gemini:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY missing. Add your key to .env."
            )

        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _wait_for_rate_limit(self):
        global _last_request_at

        elapsed = time.monotonic() - _last_request_at
        wait = MIN_SECONDS_BETWEEN_REQUESTS - elapsed

        if wait > 0:
            print(f"Rate-limit pause: {wait:.1f}s")
            time.sleep(wait)

        _last_request_at = time.monotonic()

    @staticmethod
    def _is_daily_quota_error(exc):
        message = str(exc).casefold()

        daily_indicators = (
            "per day",
            "per_day",
            "requestsperday",
            "generate_requests_per_day",
            "daily quota",
            "daily limit",
        )

        return any(term in message for term in daily_indicators)

    @staticmethod
    def _is_retryable_error(exc):
        message = str(exc).casefold()

        return any(
            term in message
            for term in (
                "429",
                "resource_exhausted",
                "rate limit",
                "503",
                "unavailable",
                "500",
                "502",
                "504",
                "internal server error",
                "deadline exceeded",
            )
        )

    @staticmethod
    def _retry_delay(exc, attempt):
        message = str(exc)

        # Use Gemini's suggested retry delay when provided.
        match = re.search(
            r"""retryDelay["']?\s*:\s*["']?(\d+(?:\.\d+)?)s""",
            message,
            re.IGNORECASE,
        )

        if match:
            return max(5.0, float(match.group(1)) + 1.0)

        # Otherwise use exponential backoff.
        return min(5.0 * (2 ** attempt), 60.0)

    def _generate(self, prompt, json_mode):
        for attempt in range(MAX_RETRIES + 1):

            self._wait_for_rate_limit()

            if json_mode:
                config = types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                )
            else:
                config = types.GenerateContentConfig(
                    temperature=0.1
                )

            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=config,
                )

                return response.text or (
                    "{}" if json_mode else ""
                )

            except Exception as exc:

                if self._is_daily_quota_error(exc):
                    raise RuntimeError(
                        "Gemini daily quota appears exhausted. "
                        "Stopping instead of wasting more requests. "
                        f"Original error: {exc}"
                    ) from exc

                if not self._is_retryable_error(exc):
                    raise

                if attempt >= MAX_RETRIES:
                    raise

                delay = self._retry_delay(exc, attempt)

                print(
                    f"Temporary Gemini error. "
                    f"Retry {attempt + 1}/{MAX_RETRIES} "
                    f"in {delay:.1f} seconds."
                )

                time.sleep(delay)

        raise RuntimeError("Gemini request failed.")

    def ask_json(self, prompt, temperature=0.1):
        response_text = self._generate(
            prompt,
            json_mode=True,
        )

        try:
            return json.loads(response_text)

        except json.JSONDecodeError:
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start >= 0 and end > start:
                try:
                    return json.loads(
                        response_text[start:end + 1]
                    )
                except json.JSONDecodeError:
                    pass

            raise ValueError(
                "Gemini did not return valid JSON: "
                + response_text[:500]
            )

    def ask_text(self, prompt, temperature=0.1):
        return self._generate(
            prompt,
            json_mode=False,
        )
