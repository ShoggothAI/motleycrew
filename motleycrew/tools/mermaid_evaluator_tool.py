# https://nodejs.org/en/download
# npm install -g @mermaid-js/mermaid-cli
import io
import os.path
import subprocess
import tempfile
from typing import Optional, List

from langchain_core.pydantic_v1 import create_model, Field
from langchain_core.tools import Tool

from motleycrew.tools import MotleyTool


class MermaidEvaluatorTool(MotleyTool):
    def __init__(
        self,
        format: Optional[str] = "svg",
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        def eval_mermaid_partial(mermaid_code: str):
            return eval_mermaid(mermaid_code, format)

        langchain_tool = Tool.from_function(
            func=eval_mermaid_partial,
            name="mermaid_evaluator_tool",
            description="Evaluates Mermaid code and returns the output as a BytesIO object.",
            args_schema=create_model(
                "MermaidEvaluatorToolInput",
                mermaid_code=(str, Field(description="The Mermaid code to evaluate.")),
            ),
        )
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


def eval_mermaid(mermaid_code: str, format: Optional[str] = "svg") -> io.BytesIO:
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", suffix=".mmd") as temp_in:
        temp_in.write(mermaid_code)
        temp_in.flush()  # Ensure all data is written to disk

        if format in ["md", "markdown"]:
            raise NotImplementedError("Markdown format is not yet supported in this wrapper.")
        assert format in [
            "svg",
            "png",
            "pdf",
        ], "Invalid format specified, must be svg, png, or pdf."
        out_file = f"output.{format}"

        # Prepare the command to call the mermaid CLI
        full_code = f"mmdc -i {temp_in.name} -o {out_file} -b transparent"

        try:
            # Execute the command
            subprocess.run(
                full_code,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # If the process succeeds, read the output file into BytesIO
            with open(out_file, "rb") as f:
                output_bytes = io.BytesIO(f.read())
            return output_bytes
        except subprocess.CalledProcessError as e:
            # If the process fails, print the error message
            return e.stderr.decode()
        finally:
            # Clean up the output file if it exists
            try:
                os.remove(out_file)
            except FileNotFoundError:
                pass


# Define the Mermaid code for the flowchart

if __name__ == "__main__":
    mermaid_code = """
    graph TD;
        A[Start] --> B[Decision]
        B -- Yes --> C[Option 1]
        B -- No --> D[Option 2]
        C --> E[End]
        D --> E
        E[End] --> F[End]

        [[
    """

    out1 = eval_mermaid(mermaid_code)
    output_file_path = "output_file.bin"
    if isinstance(out1, str):
        print(out1)
        exit(1)
    # Ensure the pointer is at the beginning of the BytesIO object
    out1.seek(0)

    # Open the output file in binary write mode and write the contents of the BytesIO object
    with open(output_file_path, "wb") as file_output:
        file_output.write(out1.read())
    tool = MermaidEvaluatorTool()
    out2 = tool.invoke({"mermaid_code": mermaid_code})
    print(out2)
