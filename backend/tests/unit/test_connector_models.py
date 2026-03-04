from backend.src.models.connectors import (
    CredentialField,
    ConnectorParam,
    ConnectorAction,
    ConnectorInfo,
    ConnectorListResponse,
    ConnectorConfigUpdate,
    ConnectorInvokeRequest,
)


def test_credential_field_secret_default():
    f = CredentialField(name="api_key", label="API Key")
    assert f.secret is True


def test_connector_action_has_params():
    a = ConnectorAction(
        name="send_email",
        description="Send email",
        params=[ConnectorParam(name="to", description="Recipient", required=True)],
    )
    assert len(a.params) == 1
    assert a.params[0].required is True


def test_connector_info_enabled_default_false():
    info = ConnectorInfo(
        name="mailgun",
        display_name="Mailgun",
        description="Email via Mailgun",
        credential_fields=[],
        actions=[],
    )
    assert info.enabled is False
    assert info.configured is False


def test_connector_invoke_request_params_optional():
    req = ConnectorInvokeRequest(action="send_email")
    assert req.params == {}
