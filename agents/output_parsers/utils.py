from langchain_core.utils.function_calling import convert_to_openai_function
from tools.tools import tools
import re
import ast

def parse_args(args: str):
    args = args.strip()
    args = args.replace("true", "True")
    args = args.replace("false", "False")
    args = args.replace("null", "None")
    args = args.replace("\"", "\"\"\"")
    i = 0
    while args[i] != "\"" and args[i] != "\'" and i < len(args) - 1:
        i += 1
    args = args[i:]
    if args[-4:] != "True" and args[-5:] != "False":
        i = len(args) - 1
        while args[i] != "\"" and args[i] != "\'" and i > 0:
            i -= 1
        args = args[:i + 1]
    print(args)
    return ast.literal_eval("{" + args + "}")

def parse_tool_call(call: str):
    call = call.strip()
    name: bool = "\"name\": " in call or "\'name\':" in call
    args: bool = "\"arguments\": " in call or "\'arguments\':" in call
    if not name:
        print({"arguments": {}, "name": "missing_function_call"})
        return {"arguments": {}, "name": "missing_function_call"}
    if not args:
        pattern = re.compile(r"\"name\": \"(.*?)\"|\'name\': \'(.*?)\'", re.DOTALL)
        match = pattern.findall(call)
        for n in match:
            if isinstance(n, tuple):
                n = n[0]
            print({"arguments": {}, "name": n})
            return {"arguments": {}, "name": n}
    args_pattern = re.compile(r"\"arguments\": {(.*?)}|\'arguments\': {(.*?)}", re.DOTALL)
    args_match = args_pattern.findall(call)
    for a in args_match:
        print(a, "\n")
        print(a[0])
        args = parse_args(a[0])
    name_pattern = re.compile(r"\"name\": \"(.*?)\"", re.DOTALL)
    name_match = name_pattern.findall(call)
    for n in name_match:
        if isinstance(n, tuple):
            n = n[0]
        print({"arguments": args, "name": n})
        return {"arguments": args, "name": n}

    
def check_tool_call(call: dict):
    global tools
    tools = [convert_to_openai_function(t) for t in tools]
    if call["name"] not in [t["name"] for t in tools]:
        return "handle_tools_error", {"error": {"error": {"name": call["name"]}}}
    tool = next((t for t in tools if t["name"] == call["name"]), None)

    if set(list(tool["parameters"]["properties"])) != set(list(call["arguments"])):
        print({"tool_response": {"error": {"expected": list(tool["parameters"]["properties"]), "received": list(call["arguments"])}, "name": call["name"]}})
        return "handle_tools_error", {"error": {"error": {"expected": list(tool["parameters"]["properties"]), "received": list(call["arguments"])}, "name": call["name"]}}
    return call["name"], call["arguments"]