from langchain.agents import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def handle_tools_error(error: dict, **kwargs) -> str:
    """This tool handles errors when the expected arguments for a tool does not match the received arguments."""
    if error["error"].get("name"):
        tool = error["error"]["name"]
        return f"Error: The tool '{tool}' does not exist. Please check the tool's name and try again."
    tool = error["name"]
    expected = error["error"]["expected"]
    received = error["error"]["received"]
    return f"Error: The expected arguments for the tool '{tool}' are {expected}, but the received arguments are {received}. Please check the tool's arguments and try again."
    
@tool
def save_file(input: str, file_name: str, append: bool = True) -> str:
    """This tool saves the input data to a file."""  
    append = "a" if append else "w"
    with open(file_name, append) as file:
        file.write(input)
    return f"File '{file_name}' saved successfully."

@tool
def get_word_length(input: str) -> int:
    """Returns the length of a word."""
    return len(input)

@tool
def web_search(query: str) -> str:
    """Useful to search the internet about a given topic and return relevant results"""
    duckduckgo_search = DuckDuckGoSearchRun()
    return duckduckgo_search(query)

tools = [handle_tools_error, get_word_length, save_file, web_search]