from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


def public_camera_name(source: object) -> str:
    """Return a useful camera label without credentials or URL query secrets."""
    text = str(source)
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
        host = parts.hostname or "camera"
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except ValueError:
        # Malformed URLs still must not expose user-info.
        scheme, remainder = text.split("://", 1)
        authority, separator, path = remainder.partition("/")
        authority = authority.rsplit("@", 1)[-1]
        return f"{scheme}://{authority}{separator}{path.split('?', 1)[0].split('#', 1)[0]}"
