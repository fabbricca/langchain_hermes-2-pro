from langchain_community.chat_models import ChatOllama
from prompts.prompt import nous_hermes_prompt
from agents.nous_hermes_functions_agent.base import create_nous_hermes_functions_agent
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor
from tools.tools import tools
from langchain.memory import ChatMessageHistory

llm = ChatOllama(model = "adrienbrault/nous-hermes2pro:Q5_K_S", temperature = 0.55)

tools_dict = [convert_to_openai_function(t) for t in tools]

history = ChatMessageHistory()

agent = create_nous_hermes_functions_agent(llm=llm, prompt=nous_hermes_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
while True:
    try:
        inp = input("User:")
        if inp == "/bye":
            break

        response = agent_executor.invoke({"input": inp, "chat_history": history, "tools" : tools_dict})
        response['output'] = response['output'].replace("<|im_end|>", "")
        history.add_user_message(inp)
        history.add_ai_message(response['output'])

        print(response['output'])
    except Exception as e:
        print(e)
        continue