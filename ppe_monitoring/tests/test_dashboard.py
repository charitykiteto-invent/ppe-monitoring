import asyncio

import httpx

from ppe_monitoring.dashboard.server import DashboardState, create_app
from ppe_monitoring.storage.event_repository import EventRepository


def test_dashboard_api_responses(tmp_path):
    state = DashboardState()
    repository = EventRepository(tmp_path / "events.db")
    app = create_app(state, repository, lambda: True, lambda: True)
    async def verify():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            live = await client.get("/")
            analytics = await client.get("/analytics")
            assert live.status_code == 200
            assert "Worksite safety at a glance" in live.text
            assert 'id="theme-toggle"' in live.text
            assert analytics.status_code == 200
            assert "Compliance rate by minute" in analytics.text
            assert "Helmet roles recorded" in analytics.text
            response = await client.get("/api/status")
            assert response.status_code == 200
            assert response.json()["summary"]["total"] == 0
            assert (await client.get("/api/events")).json() == {"events": []}
            assert (await client.post("/api/monitoring/start")).status_code == 202

    asyncio.run(verify())
