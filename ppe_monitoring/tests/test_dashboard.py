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
            assert (await client.get("/")).status_code == 200
            response = await client.get("/api/status")
            assert response.status_code == 200
            assert response.json()["summary"]["total"] == 0
            assert (await client.get("/api/events")).json() == {"events": []}
            assert (await client.post("/api/monitoring/start")).status_code == 202

    asyncio.run(verify())
