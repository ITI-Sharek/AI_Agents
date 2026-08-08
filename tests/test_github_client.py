from sharek_agents.shared_tools.github_client import GithubClient


def test_public_client_omits_empty_authorization_header(monkeypatch) -> None:
    monkeypatch.setattr(
        "sharek_agents.shared_tools.github_client.settings.github_token",
        "",
    )

    client = GithubClient()

    assert "Authorization" not in client._headers


def test_authenticated_client_sends_bearer_token() -> None:
    client = GithubClient(token="test-token")

    assert client._headers["Authorization"] == "Bearer test-token"
