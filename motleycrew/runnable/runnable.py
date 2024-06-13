from typing import Optional, Any, List, Type
from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable, RunnableSequence, RunnableConfig
from langchain_core.runnables.base import RunnableLike
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.utils import Input, Output

from motleycrew.common import logger
from motleycrew.common.exceptions import RunnableSchemaMismatch


def _check_pydantic_model_has_fields_defined(model: Type[BaseModel]) -> bool:
    """Check if a Pydantic model has fields defined besides the root."""
    return set(model.__fields__.keys()) != {"__root__"}


def _check_pydantic_field_is_required(field_info: Any) -> bool:
    """
    Check if a Pydantic model field is required.
    Made for compatibility with Pydantic v1 and v2.
    """
    if hasattr(field_info, "is_required"):  # v2
        return field_info.is_required()
    return field_info.required  # v1


class MotleyRunnable(Runnable[Input, Output], ABC):
    _input_schema: Optional[Type[BaseModel]] = None
    _output_schema: Optional[Type[BaseModel]] = None

    enforce_input_schema: bool = True
    enforce_output_schema: bool = True

    @property
    def input_schema(self) -> Type[BaseModel]:
        if self._input_schema is None:
            return super().input_schema
        return self._input_schema

    @input_schema.setter
    def input_schema(self, schema: Type[BaseModel]) -> None:
        self._input_schema = schema

    @property
    def has_input_schema(self) -> bool:
        """Check if the runnable has an input schema that is not a passthrough schema.

        Returns:
            bool:
        """
        return self.input_schema is not None and _check_pydantic_model_has_fields_defined(
            self.input_schema
        )

    @property
    def output_schema(self) -> Type[BaseModel]:
        if self._output_schema is None:
            return super().output_schema
        return self._output_schema

    @output_schema.setter
    def output_schema(self, schema: Type[BaseModel]) -> None:
        self._output_schema = schema

    @property
    def has_output_schema(self) -> bool:
        """Check if the runnable has an output schema that is not a passthrough schema.

        Returns:
            bool:
        """
        return self.output_schema is not None and _check_pydantic_model_has_fields_defined(
            self.output_schema
        )

    @abstractmethod
    def _invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Transform a single input into an output. Override to implement.

        Args:
            input: The input to the runnable.
            config: A config to use when invoking the runnable.
               Refer to Langchain Runnable class documentation for more info.

        Returns:
            The output of the runnable.
        """

    def _prepare_input_from_args(
        self, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[Input, Optional[RunnableConfig]]:
        """Prepare input and config for the runnable from args and kwargs.

        Args:
            args: The positional arguments to the runnable.
            kwargs: The keyword arguments to the runnable.

        Returns:
            A tuple of input and config.
        """
        config = kwargs.pop("config", None)

        if kwargs:
            if args:
                if len(args) == 1 and isinstance(args[0], dict):
                    config = args[0]
                else:
                    raise ValueError("Cannot pass both args and kwargs to invoke")
            return kwargs, config

        if len(args) > 2:
            raise ValueError("Cannot pass more than two positional arguments to invoke")

        if not args:
            return {}, config

        arg = args[0]

        if len(args) == 2:
            if not isinstance(args[1], dict):
                raise ValueError("Second positional argument to invoke must be a config")
            config = args[1]

        if isinstance(arg, dict):
            return arg, config

        if not self.has_input_schema:
            return arg, config

        if len(self.input_schema.__fields__) > 1:
            raise ValueError(
                "Cannot pass a single non-dict positional argument to invoke "
                "when input schema has multiple fields"
            )

        return {list(self.input_schema.__fields__.keys())[0]: arg}, config

    def _filter_input(self, input: Input) -> Input:
        """Filter input according to the schema.

        Args:
            input: The input to the runnable.

        Returns:
            The filtered input.
        """
        if not self.has_input_schema:
            return input

        filtered_input = {
            key: value for key, value in input.items() if key in self.input_schema.__fields__
        }
        return filtered_input

    def invoke(self, *args, **kwargs) -> Output:
        """Public method for invoking the runnable.
        Calls _invoke, validating its input and output according to schema, if specified.

        Args:
            *args: The positional arguments to the runnable.
            **kwargs: The keyword arguments to the runnable.

        Returns:
            The output of the runnable.
        """
        logger.debug(f"Invoking {self} with args: {args}, kwargs: {kwargs}")
        input, config = self._prepare_input_from_args(args, kwargs)
        logger.debug(f"Prepared input: {input}")

        input = self._filter_input(input)

        if self.enforce_input_schema:
            self.input_schema.parse_obj(input)
        output = self._invoke(input, config=config)
        if self.enforce_output_schema:
            self.output_schema.parse_obj(output)

        return output


RunnableSequence__init__ = RunnableSequence.__init__


def _runnable_sequence_init_with_validation(
    self: RunnableSequence,
    *steps: RunnableLike,
    name: Optional[str] = None,
    first: Optional[Runnable[Any, Any]] = None,
    middle: Optional[List[Runnable[Any, Any]]] = None,
    last: Optional[Runnable[Any, Any]] = None,
) -> None:
    for step, prev_step in zip(steps[1:], steps[:-1]):
        if (
            (isinstance(step, MotleyRunnable) or isinstance(prev_step, MotleyRunnable))
            and _check_pydantic_model_has_fields_defined(step.input_schema)
            and _check_pydantic_model_has_fields_defined(prev_step.output_schema)
        ):
            for field, field_info in step.input_schema.__fields__.items():
                if _check_pydantic_field_is_required(field_info):
                    if field not in prev_step.output_schema.__fields__:
                        raise RunnableSchemaMismatch(
                            f"Field {field} is required by {step} but not provided by {prev_step}"
                        )

    RunnableSequence__init__(
        self,
        *steps,
        name=name,
        first=first,
        middle=middle,
        last=last,
    )


RunnableSequence.__init__ = _runnable_sequence_init_with_validation
