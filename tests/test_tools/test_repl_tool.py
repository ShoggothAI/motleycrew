from motleycrew.tools.code import PythonREPLTool


class TestREPLTool:
    def test_repl_tool(self):
        repl_tool = PythonREPLTool()
        repl_tool_input_fields = list(repl_tool.tool.args_schema.__fields__.keys())

        assert repl_tool_input_fields == ["command"]
        assert repl_tool.invoke({repl_tool_input_fields[0]: "print(1)"}).strip() == "1"
