"""
The module contains callback handlers for sending data to the Lunary service
"""

from typing import List, Dict, Optional, Any, Union
import traceback

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    from llama_index.core.callbacks.schema import CBEventType, EventPayload
    from llama_index.core.base.llms.types import ChatMessage
except ImportError:
    BaseCallbackHandler = object
    CBEventType = None
    ChatMessage = None

try:
    from lunary import track_event, event_queue_ctx
except ImportError:
    track_event = None
    event_queue_ctx = None

from motleycrew.common.enums import LunaryRunType, LunaryEventName
from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.common import logger


def event_delegate_decorator(f):
    """Decorator of llamaindex event handlers

    It is used to transfer event data to a specific handler

    Args:
        f (callable):
    """
    def wrapper(self, *args, **kwargs):
        ensure_module_is_installed("llama_index")
        run_type = "start" if "start" in f.__name__ else "end"

        # find event type
        if args:
            event_type = args[0]
        else:
            event_type = kwargs.get("event_type", None)

        if not [event for event in CBEventType if event.value == event_type.value]:
            return f(*args, **kwargs)

        # find and call handler
        handler_name = "_on_{}_{}".format(event_type.value, run_type)
        handler = getattr(self, handler_name, None)
        if handler is not None and callable(handler):
            handler_args = list(args)[1:]
            try:
                event_params = handler(*handler_args, **kwargs)
                run_id = event_params.get("run_id")
                run_type = event_params.get("run_type")
                self._track_event(**event_params)
                self._event_run_type_ids.append((run_type, run_id))
            except Exception as e:
                msg = "[Lunary] An error occurred in {}: {}\n{}".format(
                    handler_name, e, traceback.format_exc()
                )
                logger.warning(msg)
        else:
            msg = "No handler with the name of the {} was found for {}".format(
                handler_name, self.__class__
            )
            logger.warning(msg)

        return f(*args, **kwargs)

    return wrapper


def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
    """Creates and returns a dict based on the message

    Args (ChatMessage): llamaindex chat message

    Returns:
        dict: dict chat message data
    """
    keys = ["function_call", "tool_calls", "tool_call_id", "name"]
    output = {"content": message.content, "role": message.role.value}
    output.update(
        {
            key: message.additional_kwargs.get(key)
            for key in keys
            if message.additional_kwargs.get(key) is not None
        }
    )
    return output


class LlamaIndexLunaryCallbackHandler(BaseCallbackHandler):

    AGENT_NAME = "LlamaIndexAgent"

    def __init__(
        self,
        app_id: str,
        event_starts_to_ignore: List[CBEventType] = None,
        event_ends_to_ignore: List[CBEventType] = None,
    ):
        """
        Lamaindex event handler class for sending data to lunary

        Args:
            app_id (str): lunary app id
            event_starts_to_ignore (List[CBEventType]): List of events for which the start of the event is ignored
            event_ends_to_ignore (List[CBEventType]): List of events for which event completion processing is ignored
        """
        ensure_module_is_installed("llama_index")
        ensure_module_is_installed("lunary")
        super(LlamaIndexLunaryCallbackHandler, self).__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or [],
        )

        self.__app_id = app_id
        self._track_event = track_event
        self._event_run_type_ids = []

        self.queue = event_queue_ctx.get()

    def _get_initial_track_event_params(
        self, run_type: LunaryRunType, event_name: LunaryEventName, run_id: str = None
    ) -> dict:
        """Return initial params for track event

        Args:
            run_type (LunaryRunType): event run type (llm, agent, tool, chain, embed)
            event_name (LunaryEventName): event name (start, end, update, error)

        Returns:
            dict:
        """
        params = {
            "run_type": run_type,
            "event_name": event_name,
            "app_id": self.__app_id,
            "callback_queue": self.queue,
        }
        if run_id is not None:
            params["run_id"] = run_id
        return params

    def _on_llm_start(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Llm start event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            parent_id (str): parent event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        messages = payload.get(EventPayload.MESSAGES)
        serialized = payload.get(EventPayload.SERIALIZED)

        params = self._get_initial_track_event_params(
            LunaryRunType.LLM, LunaryEventName.START, event_id
        )
        params["name"] = serialized.get("model")

        tag = serialized.get("class_name")
        if tag is not None:
            params["tags"] = [tag]

        params["parent_run_id"] = self.check_parent_id(parent_id)
        params["input"] = [_message_to_dict(message) for message in messages]
        return params

    def _on_llm_end(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Llm end event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        response = payload.get(EventPayload.RESPONSE)

        params = self._get_initial_track_event_params(
            LunaryRunType.LLM, LunaryEventName.END, event_id
        )
        params["output"] = _message_to_dict(response.message)
        return params

    def _on_function_call_start(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Tool start event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            parent_id (str): parent event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        tool = payload.get(EventPayload.TOOL)
        function_call = payload.get(EventPayload.FUNCTION_CALL)

        params = self._get_initial_track_event_params(
            LunaryRunType.TOOL, LunaryEventName.START, event_id
        )

        params["parent_run_id"] = self.check_parent_id(parent_id)
        params["name"] = tool.name

        params_inputs = []
        for k, v in function_call.items():
            params_inputs.append('{{"{}":"{}"}}'.format(k, v))
        params["input"] = "\n".join(params_inputs)

        return params

    def _on_function_call_end(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Tool end event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        params = self._get_initial_track_event_params(
            LunaryRunType.TOOL, LunaryEventName.END, event_id
        )
        params["output"] = payload.get(EventPayload.FUNCTION_OUTPUT) if payload is not None else ""
        return params

    def _on_agent_step_start(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Agent start event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            parent_id (str): parent event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        messages = payload.get(EventPayload.MESSAGES)

        params = self._get_initial_track_event_params(
            LunaryRunType.AGENT, LunaryEventName.START, event_id
        )
        params["parent_run_id"] = self.check_parent_id(parent_id)
        params["input"] = "\n".join(messages)
        params["name"] = self.AGENT_NAME

        return params

    def _on_agent_step_end(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Llm end event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """
        params = self._get_initial_track_event_params(
            LunaryRunType.AGENT, LunaryEventName.END, event_id
        )

        if payload:
            response = payload.get(EventPayload.RESPONSE)
            output  = response.response
        else:
            output = ""
        params["output"] = output

        return params

    def _on_exception_start(
        self,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> dict:
        """Exception start event handler

        Args:
            payload (dict): related event data
            event_id (str): event id (uuid)
            parent_id (str): parent event id (uuid)
            **kwargs:

        Returns:
            dict: dictionary of data to send to lunary
        """

        if self._event_run_type_ids:
            run_type, run_id = self._event_run_type_ids[-1]
        else:
            run_type = LunaryRunType.AGENT
            parent_id = self.check_parent_id(parent_id)
            run_id = parent_id if parent_id else event_id

        _exception = payload.get(EventPayload.EXCEPTION)
        params = self._get_initial_track_event_params(run_type, LunaryEventName.ERROR, run_id)
        params["error"] = {"message": str(_exception), "stack": traceback.format_exc()}
        return params

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass

    @event_delegate_decorator
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Start event handler

        Args:
            event_type (CBEventType): llamaindex event type
            payload (dict): related event data
            event_id (str): event id (uuid)
            parent_id (str): parent event id (uuid)
            **kwargs:

        Returns:
            str: event id (uuid)
        """
        return event_id

    @event_delegate_decorator
    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Start event handler

        Args:
            event_type (CBEventType): llamaindex event type
            payload (dict): related event data
            event_id (str): event id (uuid)
            **kwargs:
       """
        return

    @staticmethod
    def check_parent_id(parent_id: str) -> Union[str, None]:
        """Checking the occurrence of a value in the ignored list

        Args:
            parent_id (str): parent id (uuid)

        Returns:
            str: return parent id or None
        """
        if parent_id in ["root"]:
            return None
        return parent_id
