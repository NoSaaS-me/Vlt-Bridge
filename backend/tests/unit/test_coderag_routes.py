"""Tests for CodeRAG API routes.

Tests the /api/coderag/* endpoints for code index management and status tracking.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from backend.src.api.main import app
from backend.src.api.middleware import AuthContext, get_auth_context
from backend.src.api.routes.coderag import get_oracle_bridge
from backend.src.services.oracle_bridge import OracleBridge

client = TestClient(app)


@pytest.fixture
def mock_auth():
    """Provide a mock auth context for tests."""
    mock = Mock(spec=AuthContext)
    mock.user_id = "test-user"
    return mock


@pytest.fixture
def mock_oracle_bridge():
    """Provide a mock OracleBridge for tests."""
    mock_bridge = Mock(spec=OracleBridge)
    return mock_bridge


@pytest.fixture
def setup_overrides(mock_auth, mock_oracle_bridge):
    """Set up dependency overrides for testing."""
    app.dependency_overrides[get_auth_context] = lambda: mock_auth
    app.dependency_overrides[get_oracle_bridge] = lambda: mock_oracle_bridge
    yield
    app.dependency_overrides = {}


class TestGetCodeRAGStatus:
    """Tests for GET /api/coderag/status endpoint."""

    def test_status_success_ready(self, mock_oracle_bridge, setup_overrides):
        """Test successful status retrieval for a ready index."""
        mock_oracle_bridge.get_coderag_status.return_value = {
            "project_id": "test-project",
            "status": "ready",
            "file_count": 100,
            "chunk_count": 500,
            "last_indexed_at": "2026-01-01T12:00:00Z",
            "error_message": None,
            "active_job": None,
        }

        response = client.get("/api/coderag/status?project_id=test-project")

        assert response.status_code == 200
        data = response.json()
        assert data["project_id"] == "test-project"
        assert data["status"] == "ready"
        assert data["file_count"] == 100
        assert data["chunk_count"] == 500
        assert data["active_job"] is None

    def test_status_not_initialized(self, mock_oracle_bridge, setup_overrides):
        """Test status for project without code index."""
        mock_oracle_bridge.get_coderag_status.return_value = {
            "error": True,
            "message": "Project not found or index not initialized",
        }

        response = client.get("/api/coderag/status?project_id=new-project")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_initialized"
        assert data["file_count"] == 0
        assert data["chunk_count"] == 0

    def test_status_with_active_job(self, mock_oracle_bridge, setup_overrides):
        """Test status showing active indexing job."""
        mock_oracle_bridge.get_coderag_status.return_value = {
            "project_id": "test-project",
            "status": "indexing",
            "file_count": 50,
            "chunk_count": 200,
            "last_indexed_at": None,
            "error_message": None,
            "active_job": {
                "job_id": "job-123",
                "progress_percent": 45,
                "files_processed": 50,
                "files_total": 100,
                "started_at": "2026-01-01T12:00:00Z",
            },
        }

        response = client.get("/api/coderag/status?project_id=test-project")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexing"
        assert data["active_job"] is not None
        assert data["active_job"]["job_id"] == "job-123"
        assert data["active_job"]["progress_percent"] == 45

    def test_status_missing_project_id(self, setup_overrides):
        """Test status without required project_id parameter."""
        response = client.get("/api/coderag/status")

        assert response.status_code == 422  # Validation error


class TestInitCodeRAG:
    """Tests for POST /api/coderag/init endpoint."""

    def test_init_success(self, mock_oracle_bridge, setup_overrides):
        """Test successful CodeRAG initialization."""
        mock_oracle_bridge.init_coderag.return_value = {
            "job_id": "job-456",
            "status": "queued",
            "message": "Indexing job queued successfully",
        }

        response = client.post(
            "/api/coderag/init",
            json={
                "project_id": "test-project",
                "target_path": "/path/to/code",
                "force": False,
                "background": True,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["job_id"] == "job-456"
        assert data["status"] == "queued"
        assert "message" in data

    def test_init_with_force(self, mock_oracle_bridge, setup_overrides):
        """Test initialization with force flag."""
        mock_oracle_bridge.init_coderag.return_value = {
            "job_id": "job-789",
            "status": "queued",
            "message": "Re-indexing job queued (force=true)",
        }

        response = client.post(
            "/api/coderag/init",
            json={
                "project_id": "test-project",
                "target_path": "/path/to/code",
                "force": True,
                "background": True,
            },
        )

        assert response.status_code == 201
        # Verify force flag was passed
        call_args = mock_oracle_bridge.init_coderag.call_args
        assert call_args.kwargs.get("force") is True or call_args[1].get("force") is True

    def test_init_index_exists_conflict(self, mock_oracle_bridge, setup_overrides):
        """Test initialization when index already exists."""
        mock_oracle_bridge.init_coderag.return_value = {
            "error": True,
            "message": "Index already exists for project. Use force=true to re-index.",
        }

        response = client.post(
            "/api/coderag/init",
            json={
                "project_id": "test-project",
                "target_path": "/path/to/code",
            },
        )

        assert response.status_code == 409

    def test_init_invalid_path(self, mock_oracle_bridge, setup_overrides):
        """Test initialization with invalid path."""
        mock_oracle_bridge.init_coderag.return_value = {
            "error": True,
            "message": "Path not found: /invalid/path",
        }

        response = client.post(
            "/api/coderag/init",
            json={
                "project_id": "test-project",
                "target_path": "/invalid/path",
            },
        )

        assert response.status_code == 400


class TestGetJobStatus:
    """Tests for GET /api/coderag/jobs/{job_id} endpoint."""

    def test_job_status_running(self, mock_oracle_bridge, setup_overrides):
        """Test getting status of a running job."""
        mock_oracle_bridge.get_job_status.return_value = {
            "job_id": "job-123",
            "project_id": "test-project",
            "status": "running",
            "progress_percent": 65,
            "files_total": 100,
            "files_processed": 65,
            "chunks_created": 300,
            "started_at": "2026-01-01T12:00:00Z",
            "completed_at": None,
            "error_message": None,
            "duration_seconds": 120.5,
        }

        response = client.get("/api/coderag/jobs/job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "running"
        assert data["progress_percent"] == 65
        assert data["files_processed"] == 65

    def test_job_status_completed(self, mock_oracle_bridge, setup_overrides):
        """Test getting status of a completed job."""
        mock_oracle_bridge.get_job_status.return_value = {
            "job_id": "job-456",
            "project_id": "test-project",
            "status": "completed",
            "progress_percent": 100,
            "files_total": 100,
            "files_processed": 100,
            "chunks_created": 500,
            "started_at": "2026-01-01T12:00:00Z",
            "completed_at": "2026-01-01T12:05:00Z",
            "error_message": None,
            "duration_seconds": 300.0,
        }

        response = client.get("/api/coderag/jobs/job-456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress_percent"] == 100
        assert data["completed_at"] is not None

    def test_job_status_failed(self, mock_oracle_bridge, setup_overrides):
        """Test getting status of a failed job."""
        mock_oracle_bridge.get_job_status.return_value = {
            "job_id": "job-789",
            "project_id": "test-project",
            "status": "failed",
            "progress_percent": 30,
            "files_total": 100,
            "files_processed": 30,
            "chunks_created": 100,
            "started_at": "2026-01-01T12:00:00Z",
            "completed_at": "2026-01-01T12:02:00Z",
            "error_message": "Disk space exhausted",
            "duration_seconds": 120.0,
        }

        response = client.get("/api/coderag/jobs/job-789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "Disk space exhausted"

    def test_job_status_not_found(self, mock_oracle_bridge, setup_overrides):
        """Test getting status of non-existent job."""
        mock_oracle_bridge.get_job_status.return_value = {
            "error": True,
            "message": "Job not found: invalid-job-id",
        }

        response = client.get("/api/coderag/jobs/invalid-job-id")

        assert response.status_code == 404


class TestCancelJob:
    """Tests for POST /api/coderag/jobs/{job_id}/cancel endpoint."""

    def test_cancel_success(self, mock_oracle_bridge, setup_overrides):
        """Test successfully cancelling a job."""
        mock_oracle_bridge.cancel_job.return_value = {
            "status": "cancelled",
            "message": "Job job-123 has been cancelled",
        }

        response = client.post("/api/coderag/jobs/job-123/cancel")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_not_found(self, mock_oracle_bridge, setup_overrides):
        """Test cancelling a non-existent job."""
        mock_oracle_bridge.cancel_job.return_value = {
            "error": True,
            "message": "Job not found: invalid-job-id",
        }

        response = client.post("/api/coderag/jobs/invalid-job-id/cancel")

        assert response.status_code == 404

    def test_cancel_already_completed(self, mock_oracle_bridge, setup_overrides):
        """Test cancelling an already completed job."""
        mock_oracle_bridge.cancel_job.return_value = {
            "error": True,
            "message": "Job cannot be cancelled: already completed",
        }

        response = client.post("/api/coderag/jobs/job-456/cancel")

        assert response.status_code == 400


class TestOracleBridgeMethods:
    """Tests for OracleBridge CodeRAG methods."""

    def test_get_coderag_status_calls_vlt(self):
        """Test that get_coderag_status calls vlt CLI correctly."""
        with patch.object(OracleBridge, "_run_vlt_command") as mock_run:
            mock_run.return_value = {"status": "ready"}
            bridge = OracleBridge()

            result = bridge.get_coderag_status("my-project")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "coderag" in call_args
            assert "status" in call_args
            assert "--project" in call_args
            assert "my-project" in call_args

    def test_init_coderag_calls_vlt(self):
        """Test that init_coderag calls vlt CLI correctly."""
        with patch.object(OracleBridge, "_run_vlt_command") as mock_run:
            mock_run.return_value = {"job_id": "test-job", "status": "queued"}
            bridge = OracleBridge()

            result = bridge.init_coderag(
                project_id="my-project",
                target_path="/code/path",
                force=True,
                background=True,
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "coderag" in call_args
            assert "init" in call_args
            assert "--project" in call_args
            assert "my-project" in call_args
            assert "--path" in call_args
            assert "/code/path" in call_args
            assert "--force" in call_args
            assert "--background" in call_args

    def test_init_coderag_without_force(self):
        """Test init_coderag without force flag."""
        with patch.object(OracleBridge, "_run_vlt_command") as mock_run:
            mock_run.return_value = {"job_id": "test-job", "status": "queued"}
            bridge = OracleBridge()

            result = bridge.init_coderag(
                project_id="my-project",
                target_path="/code/path",
                force=False,
                background=True,
            )

            call_args = mock_run.call_args[0][0]
            assert "--force" not in call_args

    def test_get_job_status_calls_vlt(self):
        """Test that get_job_status calls vlt CLI correctly."""
        with patch.object(OracleBridge, "_run_vlt_command") as mock_run:
            mock_run.return_value = {"job_id": "job-123", "status": "running"}
            bridge = OracleBridge()

            result = bridge.get_job_status("job-123")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "coderag" in call_args
            assert "job" in call_args
            assert "job-123" in call_args

    def test_cancel_job_calls_vlt(self):
        """Test that cancel_job calls vlt CLI correctly."""
        with patch.object(OracleBridge, "_run_vlt_command") as mock_run:
            mock_run.return_value = {"status": "cancelled"}
            bridge = OracleBridge()

            result = bridge.cancel_job("job-123")

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "coderag" in call_args
            assert "cancel" in call_args
            assert "job-123" in call_args
