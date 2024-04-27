# https://nodejs.org/en/download
# npm install -g @mermaid-js/mermaid-cli
import os.path
from typing import Optional
import subprocess


def eval_mermaid(
    mermaid_code: str, in_file: Optional[str] = None, out_file: Optional[str] = None
):
    in_file = os.path.realpath(in_file or "mermaid_code.mmd").replace("\\", "/")
    out_file = os.path.realpath(out_file or "mermaid_graph.svg").replace("\\", "/")
    with open(in_file, "w") as f:
        f.write(mermaid_code)
    full_code = f"C:/Users/Egor/AppData/Roaming/npm/mmdc -i {in_file} -o {out_file}"
    out = subprocess.run(full_code, shell=True, check=True, capture_output=True)
    for fn in [in_file, out_file]:
        if os.path.exists(fn):
            os.remove(fn)
    return out


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
    in_file = "mermaid_code.mmd"
    out_file = "mermaid_graph.svg"

    out = eval_mermaid(mermaid_code, in_file, out_file)
    print(out)
