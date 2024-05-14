from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import yaml

with open("./prompts/template.yaml", "r") as yaml_file:
    templates = yaml.safe_load(yaml_file)

sys_msg_template: str = templates["sys_msg"]
human_msg_template: str = templates["human_msg"]

nous_hermes_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(sys_msg_template),
    HumanMessagePromptTemplate.from_template(human_msg_template),
    MessagesPlaceholder(variable_name = "agent_scratchpad")
])