import pytest

from motleycrew.crew import MotleyCrew
from motleycrew.tasks import SimpleTask


class AgentMock:
    def invoke(self, input_dict) -> str:
        clear_dict = self.clear_input_dict(input_dict)
        return str(clear_dict)

    async def ainvoke(self, input_dict) -> str:
        return self.invoke(input_dict)

    @staticmethod
    def clear_input_dict(input_dict: dict) -> dict:
        clear_dict = {}
        for param in ("name", "prompt"):
            value = input_dict.get(param, None)
            if value is not None:
                clear_dict[param] = value
        return clear_dict


class CrewFixtures:
    num_task = 0

    @pytest.fixture(scope="class")
    def crew(self):
        obj = MotleyCrew()
        return obj

    @pytest.fixture
    def agent(self):
        return AgentMock()

    @pytest.fixture
    def tasks(self, request, crew, agent):
        num_tasks = request.param or 1
        tasks = []
        for i in range(num_tasks):
            description = "task{} description".format(self.num_task)
            tasks.append(SimpleTask(description=description, agent=agent, crew=crew))
            CrewFixtures.num_task += 1
        return tasks
