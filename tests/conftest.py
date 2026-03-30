"""tests/conftest.py — Shared fixtures for all tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="session")
def client():
    """FastAPI test client — session-scoped to avoid repeated model loading."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def sample_statute_pages():
    return [
        {
            "page_num": 1,
            "raw_text": (
                "Section 302\nPunishment for murder.\n"
                "Whoever commits murder shall be punished with death, "
                "or imprisonment for life, and shall also be liable to fine.\n\n"
                "Section 304\nPunishment for culpable homicide not amounting to murder.\n"
                "Whoever commits culpable homicide not amounting to murder shall be punished "
                "with imprisonment for life, or imprisonment of either description for a term "
                "which may extend to ten years, and shall also be liable to fine."
            ),
            "source_path": "data/statutes/criminal/IPC_1860.pdf",
        }
    ]


@pytest.fixture
def sample_judgment_pages():
    return [
        {
            "page_num": 1,
            "raw_text": (
                "Kesavananda Bharati v. State of Kerala\n"
                "AIR 1973 SC 1461\n\n"
                "BENCH:\n13 Judges\n\n"
                "HEADNOTE:\nBasic structure doctrine — Parliament cannot amend the "
                "Constitution so as to destroy its basic structure.\n\n"
                "JUDGMENT:\nThis is a landmark case concerning the power of Parliament "
                "to amend the Constitution under Article 368.\n\n"
                "The question before the court was whether Parliament has unlimited "
                "constituent power to amend any provision of the Constitution.\n\n"
                "HELD:\nParliament has wide powers of amendment under Article 368 but "
                "it does not have power to abrogate or take away fundamental rights "
                "or to alter the basic structure of the Constitution."
            ),
            "source_path": "data/supreme_court_judgments/1973/kesavananda.pdf",
        }
    ]
