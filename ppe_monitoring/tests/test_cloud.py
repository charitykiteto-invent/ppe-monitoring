from ppe_monitoring.cloud.supabase_publisher import SupabasePublisher


def test_cloud_publisher_is_safely_disabled_without_service_key(monkeypatch):
    monkeypatch.delenv("PPE_TEST_SUPABASE_KEY", raising=False)
    publisher = SupabasePublisher({
        "enabled": True,
        "url": "https://example.supabase.co",
        "service_key_env": "PPE_TEST_SUPABASE_KEY",
    })
    assert not publisher.enabled
    assert "PPE_TEST_SUPABASE_KEY" in publisher.status["error"]
