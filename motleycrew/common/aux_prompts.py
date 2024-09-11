from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from motleycrew.common import MotleyTool


class AuxPrompts:
    """Singleton containing miscellaneous auxiliary prompts.
    In rare cases where you need to customize these, you can modify them before instantiating your agents.
    """

    DIRECT_OUTPUT_ERROR_WITH_SINGLE_OUTPUT_HANDLER = (
        "You must call the `{output_handler}` tool to return the final output."
    )
    DIRECT_OUTPUT_ERROR_WITH_MULTIPLE_OUTPUT_HANDLERS = (
        "You must call one of the following tools to return the final output: {output_handlers}"
    )
    AMBIGUOUS_OUTPUT_HANDLER_CALL_ERROR = (
        "You attempted to return output by calling `{current_output_handler}` tool, "
        "but included other tool calls in your response. "
        "You must only call one of the following tools to return: {output_handlers}."
    )

    @staticmethod
    def get_direct_output_error_message(output_handlers: List["MotleyTool"]) -> str:
        if len(output_handlers) == 1:
            message = AuxPrompts.DIRECT_OUTPUT_ERROR_WITH_SINGLE_OUTPUT_HANDLER.format(
                output_handler=output_handlers[0].name
            )
        else:
            message = AuxPrompts.DIRECT_OUTPUT_ERROR_WITH_MULTIPLE_OUTPUT_HANDLERS.format(
                output_handlers=", ".join([f"`{handler.name}`" for handler in output_handlers])
            )

        return message

    @staticmethod
    def get_ambiguous_output_handler_call_error_message(
        current_output_handler: "MotleyTool", output_handlers: List["MotleyTool"]
    ) -> str:
        return AuxPrompts.AMBIGUOUS_OUTPUT_HANDLER_CALL_ERROR.format(
            current_output_handler=current_output_handler.name,
            output_handlers=", ".join([f"`{handler.name}`" for handler in output_handlers]),
        )
