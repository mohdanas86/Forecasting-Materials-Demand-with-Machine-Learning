from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os

# JWT stub - in production, use proper JWT library like PyJWT
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    JWT Authentication stub

    In production, this would:
    1. Decode JWT token
    2. Validate signature
    3. Check expiration
    4. Extract user information
    5. Verify user permissions

    For now, accepts any token with 'Bearer ' prefix
    """
    token = credentials.credentials

    # Stub validation - accept any non-empty token
    if not token or len(token) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user info from token (stub)
    # In production: decode JWT and extract user_id, roles, etc.
    user_id = "user_123"  # Stub user ID

    return user_id

def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[str]:
    """
    Optional JWT authentication - allows anonymous access but extracts user if token provided
    """
    if credentials:
        return get_current_user(credentials)
    return None

# Permission checking stubs
def require_admin(user_id: str = Depends(get_current_user)) -> str:
    """Require admin permissions"""
    # Stub: assume user_123 is admin
    if user_id != "user_123":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user_id

def require_read_access(user_id: str = Depends(get_current_user)) -> str:
    """Require read access (all authenticated users)"""
    return user_id

def require_write_access(user_id: str = Depends(get_current_user)) -> str:
    """Require write access"""
    # Stub: all authenticated users have write access
    return user_id