# https://nodejs.org/en/download
# npm install -g @mermaid-js/mermaid-cli
import os.path
import subprocess
import io
import tempfile
from typing import Optional


def eval_mermaid(mermaid_code: str, format: Optional[str] = "svg") -> io.BytesIO:
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", suffix=".mmd") as temp_in:
        temp_in.write(mermaid_code)
        temp_in.flush()  # Ensure all data is written to disk

        out_file = f"output.{format}"

        # Prepare the command to call the mermaid CLI
        full_code = f"C:/Users/Egor/AppData/Roaming/npm/mmdc -i {temp_in.name} -o {out_file} -b transparent"

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
            print(f"An error occurred: {e.stderr.decode()}")
            return None
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
    """

    out = eval_mermaid(mermaid_code)
    print(out)
