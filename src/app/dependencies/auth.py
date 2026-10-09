import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import API_KEY, API_KEYS_MAP

try:
    from app.config import ADMIN_KEY
except ImportError:
    ADMIN_KEY = os.getenv("ADMIN_KEY")


security = HTTPBearer(auto_error=False)


def _get_configured_keys() -> list[str]:
    """Retrieve configured API keys dynamically to support mock patching in tests."""
    active_api_key = API_KEY
    active_keys_map = API_KEYS_MAP
    try:
        import app.main as main_mod

        active_api_key = getattr(main_mod, "API_KEY", active_api_key)
        active_keys_map = getattr(main_mod, "API_KEYS_MAP", active_keys_map)
    except ImportError:
        pass

    keys = list(active_keys_map.keys()) if active_keys_map else []
    if active_api_key and active_api_key not in keys:
        keys.append(active_api_key)
    return keys


async def verify_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[HTTPAuthorizationCredentials]:
    """
    Dependency to verify standard API Key.
    Passes through without auth if no keys are configured.
    """
    configured_keys = _get_configured_keys()
    if configured_keys:
        if auth is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Constant-time comparison to prevent timing attacks
        compare_func = secrets.compare_digest
        try:
            import app.main as main_mod

            if hasattr(main_mod, "secrets") and hasattr(
                main_mod.secrets, "compare_digest"
            ):
                compare_func = main_mod.secrets.compare_digest
        except ImportError:
            pass

        matched = False
        for valid_key in configured_keys:
            if compare_func(auth.credentials, valid_key):
                matched = True
                break

        if not matched:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return auth


async def verify_admin_key(
    auth: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[HTTPAuthorizationCredentials]:
    """
    Dependency to verify admin API Key for privileged operations.
    """
    active_admin_key = ADMIN_KEY
    try:
        import app.main as main_mod

        active_admin_key = getattr(main_mod, "ADMIN_KEY", active_admin_key)
    except ImportError:
        pass

    if active_admin_key:
        if auth is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing Admin API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not secrets.compare_digest(auth.credentials, active_admin_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing Admin API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        # Fallback to standard API Key logic if ADMIN_KEY is not configured
        return await verify_api_key(auth)

    return auth
