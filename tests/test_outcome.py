"""Tests for agentmeter.outcome — business outcome attribution."""

from agentmeter.exporters.memory import MemoryExporter
from agentmeter.outcome import record_outcome
from agentmeter.tracker import configure, reset


class TestRecordOutcome:
    def setup_method(self):
        reset()
        self.mem = MemoryExporter()
        configure(exporters=[self.mem])

    def teardown_method(self):
        reset()

    def test_basic_outcome(self):
        outcome = record_outcome(
            run_id="run-123",
            outcome="ticket_resolved",
            value_usd=12.50,
        )
        assert outcome.run_id == "run-123"
        assert outcome.outcome == "ticket_resolved"
        assert outcome.value_usd == 12.50

        # Should create an event
        assert len(self.mem.events) == 1
        event = self.mem.events[0]
        assert event.event_type == "outcome"
        assert event.run_id == "run-123"
        assert event.metadata["outcome"] == "ticket_resolved"
        assert event.metadata["value_usd"] == 12.50

    def test_outcome_without_value(self):
        outcome = record_outcome(
            run_id="run-456",
            outcome="lead_qualified",
        )
        assert outcome.value_usd is None
        event = self.mem.events[0]
        assert event.metadata["value_usd"] is None

    def test_outcome_with_metadata(self):
        outcome = record_outcome(
            run_id="run-789",
            outcome="ticket_resolved",
            value_usd=5.00,
            metadata={"ticket_id": "T-1234", "resolution_time_min": 3},
        )
        assert outcome.metadata["ticket_id"] == "T-1234"
        event = self.mem.events[0]
        assert event.metadata["ticket_id"] == "T-1234"
        assert event.metadata["resolution_time_min"] == 3

    def test_outcome_with_project_and_tags(self):
        record_outcome(
            run_id="run-abc",
            outcome="sale_completed",
            project="sales-agent",
            tags={"team": "sales"},
        )
        event = self.mem.events[0]
        assert event.project == "sales-agent"
        assert event.tags == {"team": "sales"}

    def test_outcome_has_unique_id(self):
        o1 = record_outcome(run_id="r1", outcome="a")
        o2 = record_outcome(run_id="r2", outcome="b")
        assert o1.outcome_id != o2.outcome_id

    def test_outcome_event_id_matches(self):
        outcome = record_outcome(run_id="r1", outcome="test")
        event = self.mem.events[0]
        assert event.event_id == outcome.outcome_id
