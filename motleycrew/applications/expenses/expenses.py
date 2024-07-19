from motleycrew.agents.langchain import ReActToolCallingMotleyAgent
from schema_delta import delta_generator
from sql_tools import SQLExpression

schema_critic_prompt = """ You are an experienced data engineer and you have been tasked with reviewing whether a 
SQL query correctly checks whether expenses conform to policy. You are given a schema, a query against it, and the 
expenses policy section text to verify against it. If the query correctly expresses the policy, return an empty string.
If it doesn't, call the exception tool, passing it a detailed explanation of why it doesn't.
Policy section:
{policy}
Schema:
{schema}
Query:
{query}
"""

schema_critic = ReActToolCallingMotleyAgent(
    description=schema_critic_prompt,
    input_schema="auto",  # Input schema should be auto-parsed from the prompt, with string types
    tools=[ExceptionTool],
)


class SchemaDesignerOutput(BaseModel):
    query: Field(
        SQLExpression, description="A valid PGSQL query that returns rows that violate the policy"
    )
    latest_schema: Field(
        SQLExpression,
        description="The expenses table schema, represented as a valid PGSQL query that will create the table, "
        "with any new columns you added to represent the policy",
    )


class SchemaDesignerExtendedOutput(SchemaDesignerOutput):
    data_examples: Field(
        SQLExpression,
        description="Examples of valid and invalid rows in the expenses table according to the policy",
    )


class SchemaDesignerInput(BaseModel):
    schema: Field(
        SQLExpression,
        description="The expenses table schema so far, represented as a valid PGSQL query that will create the table",
    )
    policy: Field(str, description="The policy section to be represented in the schema")


class VerifyPolicyRepresentation(MotleyTool):
    def invoke(self, latest_schema: str, query: str):
        # First check whether the inputs are valid SQL
        latest_schema = SQLExpression(latest_schema)
        query = SQLExpression(query)

        # Retrieve the inputs to the calling agent - how can we do this gracefully?
        original_schema = self.retrieve_input("schema")
        policy = self.retrieve_input("policy")

        # Now check whether the query correctly expresses the policy
        # This will raise an exception if the query doesn't correctly express the policy
        schema_critic.invoke(schema=latest_schema, query=query, policy=policy)

        # This will raise an exception if the latest schema is not a strict extension of the original one
        # Is there an easier way to do this?
        schema_change = delta_generator.invoke(
            latest_schema=SQLExpression(latest_schema),
            original_schema=SQLExpression(original_schema),
        )

        # TODO: write a data generator, separate agent with validation of output
        # For the first time in this flow, actually call postgres to verify?
        data_examples = data_example_generator.invoke(schema=latest_schema, query=query)

        return SchemaDesignerExtendedOutput(
            query=query, latest_schema=latest_schema, data_examples=data_examples
        )


# Should this be a task?
# parametrize tasks
schema_designer_prompt = """ You are an experienced data engineer and you have been tasked with designing a schema and 
a validation query to check whether expenses confirm to policy. You are given a draft schema and the description of 
a section of the expense policy. 

You are allowed to add new columns to the schema ONLY IF NECESSARY TO REPRESENT THE POLICY SECTION 
YOU ARE GIVEN. ONLY add new columns if the current policy section can't be expressed with exist

Policy section:
{policy}
Schema:
{schema}
"""

schema_designer = ReActToolCallingMotleyAgent(
    description=schema_designer_prompt,
    input_schema=SchemaDesignerInput,
    output_handler=VerifyPolicyRepresentation(),
)
