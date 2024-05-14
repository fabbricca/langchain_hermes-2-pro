from typing import Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from agents.format_scratchpad.nous_hermes_functions import (
   format_to_nous_hermes_function_messages,
)
from agents.output_parsers.nous_hermes_functions import (
   NousHermesFunctionsAgentOutputParser,
)

def create_nous_hermes_functions_agent(
   llm: BaseLanguageModel, prompt: ChatPromptTemplate
) -> Runnable:
   """Create an agent that uses Nous Hermes function calling.

   Args:
       llm: LLM to use as the agent. Should work with Nous Hermes function calling,
           so either be an Nous Hermes model that supports that or a wrapper of
           a different model that adds in equivalent support.
       prompt: The prompt to use. See Prompt section below for more.

   Returns:
       A Runnable sequence representing an agent. It takes as input all the same input
       variables as the prompt passed in does. It returns as output either an
       AgentAction or AgentFinish.
   """
   if "agent_scratchpad" not in (
       prompt.input_variables + list(prompt.partial_variables)
   ):
       raise ValueError(
           "Prompt must have input variable `agent_scratchpad`, but wasn't found."
           f"Found {prompt.input_variables} instead."
       )
   agent = (
       RunnablePassthrough.assign(
           agent_scratchpad=lambda x: format_to_nous_hermes_function_messages(
               x["intermediate_steps"]
           )
       )
       | prompt
       | llm
       | NousHermesFunctionsAgentOutputParser()
   )
   return agent