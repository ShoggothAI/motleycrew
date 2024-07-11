"""Utilities for tools"""
from typing import Type, Callable
import inspect
from copy import copy

from langchain.pydantic_v1 import BaseModel


def crewai_tool_args_decorator(args_schema: Type[BaseModel]):
    """Decorator for compatibility of tool arguments when converting from crewai_tool to langchain_tool"""
    args_schema_fields = list(args_schema.__fields__)

    def decorator(func: Callable):

        def wrapper(*args, **kwargs):
            signature = inspect.signature(func)
            signature_parameters = [k for k in signature.parameters.keys() if k != "kwargs"]

            if "args" in signature_parameters:
                return func(*args, **kwargs)

            func_kwargs = copy(kwargs)
            for schema_arg in args_schema_fields:
                if schema_arg in func_kwargs:
                    continue
                if args:
                    func_kwargs[schema_arg] = args[0]
                    args = args[1:]
                else:
                    break

            return func(**func_kwargs)

        return wrapper

    return decorator
