from typing import Optional

import pytest
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_core.runnables.utils import Input, Output
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from motleycrew.runnable import MotleyRunnable


class RunnableMock(MotleyRunnable):
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        return input


class SingleFieldModel(BaseModel):
    str_field: str = Field(description="string")


class MultipleFieldModel(BaseModel):
    str_field: str = Field(description="string")
    int_field: int = Field(description="number")


class TestRunnableSchema:
    def test_infer_input_schema_from_tool(self):
        tool = Tool.from_function(
            func=lambda input: input,
            args_schema=SingleFieldModel,
            name="SingleInputTool",
            description="Tool with single input",
        )
        runnable = MotleyRunnable(tool)
        assert runnable.input_schema == SingleFieldModel

    def test_infer_input_schema_from_prompt_template(self):
        prompt_template = PromptTemplate.from_template(
            "string: {str_field}, number: {int_field}",
        )
        runnable = MotleyRunnable(prompt_template)
        assert set(runnable.input_schema.__fields__.keys()) == {"str_field", "int_field"}

    def test_input_validation(self):
        runnable = RunnableMock()
        runnable.input_schema = SingleFieldModel

        assert runnable.invoke({"str_field": "string"}) == {"str_field": "string"}
        with pytest.raises(ValueError):
            runnable.invoke({"str_field": 1})
        with pytest.raises(ValueError):
            runnable.invoke({"extra": "string"})

    def test_input_filtering(self):
        runnable = RunnableMock()

        assert runnable.invoke({"str_field": "string"}) == {"str_field": "string"}

        runnable.input_schema = MultipleFieldModel

        assert runnable.invoke({"str_field": "string", "int_field": 1}) == {
            "str_field": "string",
            "int_field": 1,
        }
        assert runnable.invoke({"str_field": "string", "int_field": 1, "extra": "extra"}) == {
            "str_field": "string",
            "int_field": 1,
        }

    def test_input_kwargs(self):
        runnable = RunnableMock()
        assert runnable.invoke(str_field="string", extra="extra") == {
            "str_field": "string",
            "extra": "extra",
        }

        runnable.input_schema = MultipleFieldModel

        assert runnable.invoke(str_field="string", int_field=1) == {
            "str_field": "string",
            "int_field": 1,
        }
        assert runnable.invoke(str_field="string", int_field=1, extra="extra") == {
            "str_field": "string",
            "int_field": 1,
        }

    def test_input_single_value(self):
        runnable = RunnableMock()
        assert runnable.invoke("string") == "string"

        runnable.input_schema = SingleFieldModel
        assert runnable.invoke("string") == {"str_field": "string"}

    def test_error_on_invalid_input(self):
        runnable = RunnableMock()
        runnable.input_schema = MultipleFieldModel

        with pytest.raises(ValueError):
            runnable.invoke("string", 1)
        with pytest.raises(ValueError):
            runnable.invoke("string", int_field=1)
        with pytest.raises(ValueError):
            runnable.invoke({"str_field": "string"}, int_field=1)
        with pytest.raises(ValueError):
            runnable.invoke({"str_field": "string"}, 1)

    def test_output_validation(self):
        runnable = RunnableMock()
        runnable.output_schema = SingleFieldModel

        assert runnable.invoke({"str_field": "string"}) == {"str_field": "string"}
        with pytest.raises(ValueError):
            runnable.invoke({"str_field": 1})
        with pytest.raises(ValueError):
            runnable.invoke({"extra": "extra"})
        with pytest.raises(ValueError):
            runnable.invoke("string")
        with pytest.raises(ValueError):
            runnable.invoke({"str_field": "string", "extra": "extra"})  # shall we allow extras?


class TestRunnableChain:
    def test_chain_with_compatible_schemas(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_1.output_schema = MultipleFieldModel
        runnable_2.input_schema = MultipleFieldModel

        chain = runnable_1 | runnable_2
        assert chain.invoke({"str_field": "string", "int_field": 1}) == {
            "str_field": "string",
            "int_field": 1,
        }

    def test_chain_with_compatible_schemas_extra_field(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_1.output_schema = MultipleFieldModel
        runnable_2.input_schema = SingleFieldModel

        chain = runnable_1 | runnable_2
        assert chain.invoke({"str_field": "string", "int_field": 1}) == {
            "str_field": "string",
        }

    def test_chain_with_only_input_schema(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_2.input_schema = SingleFieldModel

        chain = runnable_1 | runnable_2
        assert chain.invoke({"str_field": "string", "int_field": 1}) == {
            "str_field": "string",
        }
        assert chain.invoke(str_field="string") == {
            "str_field": "string",
        }
        assert chain.invoke("string") == {
            "str_field": "string",
        }

    def test_chain_with_only_output_schema(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_1.output_schema = SingleFieldModel

        chain = runnable_1 | runnable_2
        assert chain.invoke({"str_field": "string"}) == {
            "str_field": "string",
        }

    def test_error_on_chain_with_incompatible_schemas(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_1.output_schema = SingleFieldModel
        runnable_2.input_schema = MultipleFieldModel

        with pytest.raises(TypeError):
            runnable_1 | runnable_2

    def test_chain_with_langchain_runnable(self):
        runnable = RunnableMock()
        runnable.output_schema = SingleFieldModel

        chain = runnable | RunnablePassthrough()
        assert chain.invoke({"str_field": "string"}) == {
            "str_field": "string",
        }

    def test_assign_updates_output_schema(self):
        runnable_1 = RunnableMock()
        runnable_2 = RunnableMock()
        runnable_1.output_schema = SingleFieldModel
        runnable_2.input_schema = MultipleFieldModel

        chain = runnable_1 | RunnablePassthrough.assign(int_field=lambda x: 1) | runnable_2
        assert chain.invoke({"str_field": "string"}) == {
            "str_field": "string",
            "int_field": 1,
        }
