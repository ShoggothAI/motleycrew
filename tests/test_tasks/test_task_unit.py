import pytest

from motleycrew.tasks import TaskUnit


class TestTaskUnit:

    @pytest.fixture(scope="class")
    def unit(self):
        return TaskUnit()

    def test_set_pending(self, unit):
        unit.set_pending()
        assert unit.pending

    def test_set_running(self, unit):
        unit.set_running()
        assert unit.running

    def test_set_done(self, unit):
        unit.set_done()
        assert unit.done

    def test_as_dict(self, unit):
        assert dict(unit) == unit.as_dict()
