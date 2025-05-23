from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from open_reranker.core.config import settings
from open_reranker.core.logging import setup_logging

logger = setup_logging()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()


class AuthManager:
    """Authentication and authorization manager."""

    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")


# Global auth manager instance
auth_manager = AuthManager()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> Dict[str, Any]:
    """Get the current authenticated user from JWT token."""
    if not settings.AUTH_ENABLED:
        # Return a default user when auth is disabled
        return {"sub": "anonymous", "tier": "free"}

    token = credentials.credentials
    payload = auth_manager.verify_token(token)

    if payload.get("sub") is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return payload


async def get_current_user_optional(request: Request) -> Optional[Dict[str, Any]]:
    """Get the current user if authenticated, otherwise return None."""
    if not settings.AUTH_ENABLED:
        return {"sub": "anonymous", "tier": "free"}

    # Try to get authorization header
    authorization = request.headers.get("Authorization")
    if not authorization:
        return None

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            return None

        payload = auth_manager.verify_token(token)
        if payload.get("sub") is None:
            return None

        return payload
    except (ValueError, HTTPException):
        return None


def require_tier(required_tier: str):
    """Decorator to require a specific user tier."""

    def decorator(user: Dict[str, Any] = Depends(get_current_user)):
        user_tier = user.get("tier", "free")
        tier_hierarchy = {"free": 0, "pro": 1, "enterprise": 2}

        if tier_hierarchy.get(user_tier, 0) < tier_hierarchy.get(required_tier, 0):
            raise HTTPException(
                status_code=403,
                detail=f"This feature requires {required_tier} tier or higher",
            )
        return user

    return decorator
