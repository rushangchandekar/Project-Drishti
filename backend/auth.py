"""
backend/auth.py
JWT Authentication, Password Hashing, and RBAC for Project Drishti
"""

import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from passlib.hash import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from backend.config import get_settings
from backend.database import get_db
from backend.models_db import User

settings = get_settings()
security_scheme = HTTPBearer(auto_error=False)


# ============================================================================
# Pydantic Request / Response Schemas
# ============================================================================

class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "COMMAND_OPERATOR"

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


# ============================================================================
# Password Utilities
# ============================================================================

def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt"""
    return bcrypt.hash(plain_password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash"""
    return bcrypt.verify(plain_password, hashed_password)


# ============================================================================
# JWT Token Creation & Verification
# ============================================================================

def create_access_token(user_id: str, role: str, org_id: str) -> str:
    """Create a short-lived JWT access token"""
    payload = {
        "sub": user_id,
        "role": role,
        "org_id": org_id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def create_refresh_token(user_id: str) -> str:
    """Create a long-lived JWT refresh token (7 days)"""
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    """Decode and validate a JWT token, raising HTTPException on failure"""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ============================================================================
# FastAPI Dependencies (Injectable Guards)
# ============================================================================

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Extract and validate the current user from the Authorization Bearer header.
    Returns the User ORM object or raises 401.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Bearer token.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    payload = decode_token(credentials.credentials)

    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated")

    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Optionally extract the current user. Returns None if no token is provided,
    allowing endpoints to work both authenticated and unauthenticated.
    Used during MVP transition period.
    """
    if credentials is None:
        return None
    try:
        return get_current_user(credentials, db)
    except HTTPException:
        return None


def require_roles(*allowed_roles: str):
    """
    Factory that returns a dependency requiring the user to have one of the allowed roles.
    Usage: Depends(require_roles("SUPER_ADMIN", "SAFETY_DIRECTOR"))
    """
    def role_checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        return user
    return role_checker
