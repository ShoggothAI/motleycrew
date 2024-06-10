from abc import ABC
from langchain_core.runnables import Runnable
from langchain_core.pydantic_v1 import BaseModel


class MotleyRunnable(Runnable, ABC):
    pass
