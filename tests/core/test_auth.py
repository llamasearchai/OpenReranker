import time
from datetime import timedelta
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from open_reranker.core.auth import AuthManager, pwd_context
from open_reranker.core.config import Settings


@pytest.fixture
def mock_settings():
    return Settings(SECRET_KEY="testsecret", ACCESS_TOKEN_EXPIRE_MINUTES=30)


@pytest.fixture
def auth_manager(mock_settings):
    with patch("open_reranker.core.auth.settings", mock_settings):
        return AuthManager()


class TestAuthManager:

    def test_password_hashing_and_verification(self, auth_manager):
        password = "testpassword"
        hashed_password = auth_manager.get_password_hash(password)
        assert hashed_password != password
        assert auth_manager.verify_password(password, hashed_password)
        assert not auth_manager.verify_password("wrongpassword", hashed_password)

    def test_create_access_token(self, auth_manager):
        data = {"sub": "testuser"}
        token = auth_manager.create_access_token(data)
        assert isinstance(token, str)

        decoded_payload = jwt.decode(
            token, auth_manager.secret_key, algorithms=[auth_manager.algorithm]
        )
        assert decoded_payload["sub"] == "testuser"
        assert "exp" in decoded_payload

    def test_create_access_token_with_custom_expiry(self, auth_manager):
        data = {"sub": "testuser_custom_exp"}
        expires_delta = timedelta(minutes=15)
        token = auth_manager.create_access_token(data, expires_delta=expires_delta)
        decoded_payload = jwt.decode(
            token, auth_manager.secret_key, algorithms=[auth_manager.algorithm]
        )
        expected_exp = int(time.time() + expires_delta.total_seconds())
        assert decoded_payload["exp"] == pytest.approx(
            expected_exp, abs=1
        )  # Allow 1 sec diff

    def test_verify_token_valid(self, auth_manager):
        data = {"sub": "verify_user"}
        token = auth_manager.create_access_token(data)
        payload = auth_manager.verify_token(token)
        assert payload["sub"] == "verify_user"

    def test_verify_token_expired(self, auth_manager):
        data = {"sub": "expired_user"}
        # Create a token that expires almost immediately
        token = auth_manager.create_access_token(
            data, expires_delta=timedelta(seconds=0.01)
        )
        time.sleep(0.02)  # Wait for token to expire
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token)
        assert exc_info.value.status_code == 401
        assert "Token has expired" in exc_info.value.detail

    def test_verify_token_invalid_signature(self, auth_manager):
        data = {"sub": "invalid_sig_user"}
        token = auth_manager.create_access_token(data)
        # Tamper with the token or use a wrong key
        wrong_key_manager = AuthManager()
        wrong_key_manager.secret_key = "anothersecretkey"

        with pytest.raises(HTTPException) as exc_info:
            wrong_key_manager.verify_token(
                token
            )  # Verifying token with a different key
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

        # Try verifying a token signed with another key
        token_signed_by_wrong_key = wrong_key_manager.create_access_token(data)
        with pytest.raises(HTTPException) as exc_info_2:
            auth_manager.verify_token(token_signed_by_wrong_key)
        assert exc_info_2.value.status_code == 401
        assert "Invalid token" in exc_info_2.value.detail

    def test_verify_token_malformed(self, auth_manager):
        malformed_token = "this.is.not.a.valid.token"
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(malformed_token)
        assert exc_info.value.status_code == 401
        assert "Invalid token" in exc_info.value.detail

    def test_verify_token_missing_sub(self, auth_manager):
        # "sub" is not strictly enforced by verify_token itself, but by get_current_user
        # Let's create a token without "sub" to see how verify_token handles it
        # (it should decode fine, the check is usually in the dependency using it)
        token_no_sub = jwt.encode(
            {"exp": time.time() + 3600},
            auth_manager.secret_key,
            algorithm=auth_manager.algorithm,
        )
        payload = auth_manager.verify_token(token_no_sub)
        assert "sub" not in payload
