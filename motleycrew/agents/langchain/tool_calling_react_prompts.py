from __future__ import annotations

import json

from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class ToolCallingReActPrompts:
    main_instruction: str
    output_instruction_with_output_handler: str
    output_instruction_without_output_handler: str
    example_messages: list[BaseMessage]
    reminder_message_with_output_handler: BaseMessage
    reminder_message_without_output_handler: BaseMessage

    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("system", self.main_instruction),
                MessagesPlaceholder(variable_name="example_messages", optional=True),
                MessagesPlaceholder(variable_name="input"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                MessagesPlaceholder(variable_name="additional_notes", optional=True),
            ]
        )
        self.prompt_template = self.prompt_template.partial(example_messages=self.example_messages)
        self.prompt_template_with_output_handler = self.prompt_template.partial(
            output_instruction=self.output_instruction_with_output_handler,
        )
        self.prompt_template_with_output_handler.append(self.reminder_message_with_output_handler)

        self.prompt_template_without_output_handler = self.prompt_template.partial(
            output_instruction=self.output_instruction_without_output_handler,
        )
        self.prompt_template_without_output_handler.append(
            self.reminder_message_without_output_handler
        )


class ToolCallingReActPromptsForOpenAI(ToolCallingReActPrompts):
    main_instruction = """Answer the following questions as best you can.
Think carefully, one step at a time, and outline the next step towards answering the question.

You have access to the following tools:
{tools}

To use tools, you must first describe what you think the next step should be, and then call the tool or tools to get more information.
In this case, your reply must begin with "Thought:" and describe what the next step should be, given the information so far.
The reply must contain the tool call or calls that you described in the thought.
You may include multiple tool calls in a single reply, if necessary.

If the information so far is not sufficient to answer the question precisely and completely (rather than sloppily and approximately), don't hesitate to use tools again, until sufficient information is gathered.
Don't stop this until you are certain that you have enough information to answer the question.
{output_instruction}
{output_handlers}

Begin!
"""

    output_instruction_without_output_handler = """
If you have sufficient information to answer the question, your reply must look like
```
Final Answer: [the final answer to the original input question]
```
but without the backticks."""

    output_instruction_with_output_handler = """
If you have sufficient information to answer the question, you must call the relevant tool.

NEVER return the final answer directly, but always do it by CALLING the relevant tool:
"""

    example_messages = [
        SystemMessage("Here is an example of how to use the tools:"),
        AIMessage(
            content="Thought: <your thought here>",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_aSuMulBd6JVrHCMjyKSi93na",
                        "function": {
                            "arguments": json.dumps(dict(arg_one="value_one", arg_two="value_two")),
                            "name": "tool_name",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(
            content="<tool response>",
            tool_call_id="call_aSuMulBd6JVrHCMjyKSi93na",
        ),
        AIMessage(content="OK, I will always write a thought before calling a tool."),
    ]

    reminder_message_with_output_handler = SystemMessagePromptTemplate.from_template(
        """What is the next step towards answering the question?
Your reply MUST begin with "Thought:" and describe what the next step should be.
The reply must contain the tool call or calls that you described in the thought.
TOOL CALLS WITHOUT A THOUGHT WILL NOT BE ACCEPTED!

If you have sufficient information to answer the question, call the relevant tool:
{output_handlers}

Write your reply, starting with "Thought:":
"""
    )

    reminder_message_without_output_handler = SystemMessage(
        """What is the next step towards answering the question?
Your reply MUST begin with "Thought:" and describe what the next step should be.
The reply must contain the tool call or calls that you described in the thought.
TOOL CALLS WITHOUT A THOUGHT WILL NOT BE ACCEPTED!

If you have sufficient information to answer the question, your reply must look like
```
Final Answer: [the final answer to the original input question]
```
but without the backticks. Do not include a thought with the final answer!

Write your reply, starting with either "Thought:" or "Final Answer:":
"""
    )


class ToolCallingReActPromptsForAnthropic(ToolCallingReActPrompts):
    main_instruction = """Answer the following questions as best you can.
Think carefully, one step at a time, and outline the next step towards answering the question.

You have access to the following tools:
{tools}

To use tools, you must first describe what you think the next step should be, and then call the tool or tools to get more information.
In this case, your reply must be enclosed in `<thinking></thinking>` tags and describe what the next step should be, given the information so far.
The reply must contain the tool call or calls that you described in the thought.
You may include multiple tool calls in a single reply, if necessary.

If the information so far is not sufficient to answer the question precisely and completely (rather than sloppily and approximately), don't hesitate to use tools again, until sufficient information is gathered.
Don't stop this until you are certain that you have enough information to answer the question.
{output_instruction}
{output_handlers}

Begin!
"""

    output_instruction_with_output_handler = """
If you have sufficient information to answer the question, you must call the relevant tool.

NEVER return the final answer directly, but always do it by CALLING the relevant tool:
"""

    output_instruction_without_output_handler = """
If you have sufficient information to answer the question, your reply must look like
```
<answer>
[the final answer to the original input question]
</answer>
```
but without the backticks."""

    example_messages = [
        HumanMessage("Here is an example of how to use the tools:"),
        AIMessage(
            content="""<thinking>
your thought goes here
</thinking>""",
            additional_kwargs={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_aSuMulBd6JVrHCMjyKSi93na",
                        "function": {
                            "arguments": json.dumps(dict(arg_one="value_one", arg_two="value_two")),
                            "name": "tool_name",
                        },
                        "type": "function",
                    }
                ]
            },
        ),
        ToolMessage(
            content="<tool response>",
            tool_call_id="call_aSuMulBd6JVrHCMjyKSi93na",
        ),
        AIMessage(content="OK, I will always write a thought before calling a tool."),
    ]

    reminder_message_with_output_handler = HumanMessagePromptTemplate.from_template(
        """What is the next step towards answering the question?
Your reply MUST be enclosed in `<thinking></thinking>` tags and describe what the next step should be.
The reply must contain the tool call or calls that you described in the thought.
TOOL CALLS WITHOUT A THOUGHT WILL NOT BE ACCEPTED!

If you have sufficient information to answer the question, call the relevant tool:
{output_handlers}

Write your reply, starting with `<thinking>`:
"""
    )

    reminder_message_without_output_handler = HumanMessage(
        """What is the next step towards answering the question?
Your reply MUST be enclosed in `<thinking></thinking>` tags and describe what the next step should be.
The reply must contain the tool call or calls that you described in the thought.
TOOL CALLS WITHOUT A THOUGHT WILL NOT BE ACCEPTED!

If you have sufficient information to answer the question, your reply must look like
```
<answer>
[the final answer to the original input question]
</answer>
```
but without the backticks. Do not include a thought with the final answer!

Write your reply, starting with either `<thinking>` or `<answer>`:
"""
    )
