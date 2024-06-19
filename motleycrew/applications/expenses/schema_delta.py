from sql_tools import SQLExpression


class DeltaGeneratorInput(BaseModel):
    original_schema: SQLExpression
    latest_schema: SQLExpression


delta_schema_prompt = """ You are an experienced data engineer. You are given two database schemas, the original schema
and the latest schema. The latest schema is the original schema with some columns added. 

Your job is to write a sql query that will alter the original schema to turn it into the latest schema, ONLY 
ADDING THE NEW COLUMNS. You are not allowed to modify the schema in any other way. 
If the original schema is empty, you should write a query that will create the latest schema from scratch.
If the latest schema is the same as the original schema, return a query that does nothing.
If fulfilling these instructions is not possible, call the exception tool, 
passing it a detailed explanation of why it is not possible.

(this should be prepended to the tool description of the output_handler)
ONLY return your input by calling the output handler tool provided. Your response MUST be a tool call to 
either the exception tool or the output handler tool.


Original schema:
{original_schema}

Latest schema:
{latest_schema}
"""

delta_generator = ReactToolsMotleyAgent(
    description=delta_schema_prompt,  # Should be auto-parsed into a prompt template
    input_schema=DeltaGeneratorInput,
    output_handler=SQLExpression,  # Or should it be output_tool=SQLExpression?
    tools=[ExceptionTool],  # Always include by default?
)
