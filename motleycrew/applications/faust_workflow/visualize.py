from typing import get_args, get_origin

from pyvis.network import Network

from .faust_workflow import Event, FaustWorkflow


def draw_faust_workflow(
    workflow: FaustWorkflow,
    filename: str = "faust_workflow.html",
    notebook: bool = False,
) -> None:
    """Draws the Faust workflow as a graph."""
    net = Network(directed=True, height="750px", width="100%")

    # Add the nodes + edge for stop events
    if workflow.result_event_type is not None:
        net.add_node(
            workflow.result_event_type.__name__,
            label=workflow.result_event_type.__name__,
            color="#FFA07A",
            shape="ellipse",
        )
        net.add_node("_done", label="_done", color="#ADD8E6", shape="box")
        net.add_edge(workflow.result_event_type.__name__, "_done")

    # Add nodes from all steps
    steps = {
        name: func
        for name, func in workflow.__class__.__dict__.items()
        if hasattr(func, "_is_step")
    }

    for step_name, step_func in steps.items():
        net.add_node(
            step_name, label=step_name, color="#ADD8E6", shape="box"
        )  # Light blue for steps

        # Get input and output types
        input_type = step_func.__annotations__.get("ev")
        output_types = [
            t
            for t in step_func.__annotations__.values()
            if isinstance(t, type) and issubclass(t, Event)
        ]

        if input_type:
            input_types = [input_type] if not get_origin(input_type) else get_args(input_type)
            for t in input_types:
                net.add_node(
                    t.__name__,
                    label=t.__name__,
                    color="#90EE90",
                    shape="ellipse",
                )
                net.add_edge(t.__name__, step_name)

        output_type = step_func.__annotations__.get("return")
        if output_type:
            output_types = [output_type] if not get_origin(output_type) else get_args(output_type)
            for t in output_types:
                if t != type(None):
                    net.add_node(
                        t.__name__,
                        label=t.__name__,
                        color="#90EE90" if t != workflow.result_event_type else "#FFA07A",
                        shape="ellipse",
                    )
                    net.add_edge(step_name, t.__name__)

    net.show(filename, notebook=notebook)
