"""
backend/routes_auth.py
Authentication API Router for Project Drishti
Provides /auth/register, /auth/login, /auth/refresh, /auth/me endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import User, Organization
from backend.auth import (
    RegisterRequest, LoginRequest, TokenResponse,
    hash_password, verify_password,
    create_access_token, create_refresh_token, decode_token,
    get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user account.
    Assigns user to the default 'drishti-command' organization.
    """
    # Check for existing email
    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists."
        )

    # Validate role
    valid_roles = ("SUPER_ADMIN", "COMMAND_OPERATOR", "SAFETY_DIRECTOR", "FIELD_OFFICER")
    if req.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )

    # Fetch default organization
    org = db.query(Organization).filter_by(slug="drishti-command").first()
    if not org:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Default organization not found. Run init_db() first."
        )

    # Create user
    user = User(
        organization_id=org.id,
        email=req.email,
        password_hash=hash_password(req.password),
        full_name=req.full_name,
        role=req.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Generate tokens
    access_token = create_access_token(user.id, user.role, user.organization_id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a user and return JWT access + refresh tokens.
    """
    user = db.query(User).filter(User.email == req.email).first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact your administrator."
        )

    access_token = create_access_token(user.id, user.role, user.organization_id)
    refresh_token = create_refresh_token(user.id)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=30 * 60,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role
        }
    )


@router.post("/refresh")
def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    """
    Exchange a valid refresh token for a new access token.
    """
    payload = decode_token(refresh_token)

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Expected refresh token."
        )

    user = db.query(User).filter(User.id == payload["sub"]).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or deactivated.")

    new_access = create_access_token(user.id, user.role, user.organization_id)

    return {
        "access_token": new_access,
        "token_type": "bearer",
        "expires_in": 30 * 60
    }


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user)):
    """
    Return the profile of the currently authenticated user.
    """
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
        "organization_id": current_user.organization_id,
        "created_at": current_user.created_at.strftime("%Y-%m-%dT%H:%M:%S") if current_user.created_at else None
    }
