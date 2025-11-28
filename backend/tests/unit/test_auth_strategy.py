import pytest
from unittest.mock import Mock, patch
from backend.src.services.auth import AuthService, AuthError, StaticTokenValidator, JWTValidator
from backend.src.services.config import AppConfig
from backend.src.models.auth import JWTPayload

# Mock config
@pytest.fixture
def mock_config():
    config = Mock(spec=AppConfig)
    config.enable_local_mode = True
    config.local_dev_token = "local-test"
    config.chatgpt_service_token = "gpt-secret"
    config.jwt_secret_key = "secret"
    return config

def test_static_token_validator():
    validator = StaticTokenValidator("my-secret", "test-user")
    
    # Valid token
    payload = validator.validate("my-secret")
    assert payload is not None
    assert payload.sub == "test-user"
    
    # Invalid token
    assert validator.validate("wrong") is None
    assert validator.validate("") is None

def test_auth_service_strategies(mock_config):
    auth = AuthService(config=mock_config)
    
    # Test Local Dev
    payload = auth.validate_jwt("local-test")
    assert payload.sub == "local-dev"
    
    # Test ChatGPT Service Token
    payload = auth.validate_jwt("gpt-secret")
    assert payload.sub == "demo-user"
    
    # Test Invalid
    with pytest.raises(AuthError, match="Invalid authentication credentials"):
        auth.validate_jwt("invalid-token")

def test_auth_service_priority(mock_config):
    # If both match (unlikely but possible config), first strategy wins
    # Order is Local -> ChatGPT -> JWT
    auth = AuthService(config=mock_config)
    # Strategies are added in order in __init__
    assert isinstance(auth.validators[0], StaticTokenValidator) # Local
    assert auth.validators[0].static_token == "local-test"
    assert isinstance(auth.validators[1], StaticTokenValidator) # GPT
