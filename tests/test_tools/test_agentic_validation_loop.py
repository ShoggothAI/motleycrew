import pytest
from pydantic import BaseModel
from motleycrew.tools.agentic_validation_loop import AgenticValidationLoop


class SampleSchema(BaseModel):
    x: int
    y: int


def add_ints(x: SampleSchema):
    return x.x + x.y


def test_agentic_validation_loop():
    loop = AgenticValidationLoop(
        schema=SampleSchema, name="test", description="test", post_process=add_ints
    )

    out = loop.invoke({"prompt": "Return a pair of integers"})

    assert isinstance(out, int)
    # Add more assertions based on the expected output format and values
