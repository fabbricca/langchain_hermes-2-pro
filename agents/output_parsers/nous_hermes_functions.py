import re
from json import JSONDecodeError
from typing import List, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import AgentOutputParser
from agents.output_parsers.utils import parse_tool_call, check_tool_call
import ast

class NousHermesFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with the Nous Hermes 2 Pro model, as it relies on the specific
    function_call parameter from Nous Research to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    It was add a 
    """

    @property
    def _type(self) -> str:
        return "nous-hermes-functions-agent"

    @staticmethod
    def _parse_ai_message(message: BaseMessage):
        """Parse an AI message."""
        if not isinstance(message, AIMessage):
            raise TypeError(f"Expected an AI message got {type(message)}")

        actions = []

        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        try:
            tool_calls = [parse_tool_call(t.strip()) for t in pattern.findall(message.content)]
        except:
            raise OutputParserException(
                f"Could not parse tool calls from message content: {message.content}. Please ensure that the tool calls are valid JSON."
            )

        if not tool_calls:
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )

        for tool_call in tool_calls:
            tool_name, tool_input = check_tool_call(tool_call)
            content_msg = f"\n{message.content}\n" if message.content else "\n"
            log = f"\nInvoking: `{tool_name}` with `{tool_input}`\n{content_msg}\n"
            actions.append(AgentActionMessageLog(
                tool=tool_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
            ))
        
        return actions

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")