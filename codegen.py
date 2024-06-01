from operator import itemgetter
import uuid
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph.message import AnyMessage, add_messages
from typing import Dict, TypedDict, List
from typing import Annotated
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

###### Environment Variables ######
load_dotenv()
os.environ["LANGCHAIN_PROJECT"] = "codegen"
###### End Environment Variables ######

# Select LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)

code_gen_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         You are an expert programmer. Ensure any code you provide can be executed with all
         required imports and variables defined. Structure your answer: 1) a prefix describing
         the code solution, 2) the imports, 3) the functioning code block.
         - The imports should only contain import statements. E.g. import math, import sys, etc., not actual code.
         
         - Instead of input from stdin, assume input is provided as an immutable string global variable "global_input".
         - Be sure that your solution is modular so that it's not dependent on the gloabl_input variable, that should be
         extracted in a separate function. DO NOT HARDCODE global_input in your solution, you must extract is from global global_input.
         - Likewise, instead of printing to stdout, return the result as a string global variable "global_output". 
         Storing output in global_output should also be separated from the core business logic function.
         Here is the user question:
         """
         ),
        ("placeholder", "{messages}"),
    ]
)

# Data model
# Data model


class code(BaseModel):
    """Code output"""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")
    description = "Schema for code solutions to questions about LCEL."

    def __str__(self):
        return f"Prefix: {self.prefix}\nImports: {self.imports}\nCode:\n{self.code}"


# Set to false to make more readable and save resource
code_gen_chain = code_gen_prompt_template | llm.with_structured_output(code, include_raw=True)

#question = "Write a function for fibonacci."
# messages = [("user", question)]
# result = code_gen_chain.invoke(messages)
# print(result['parsed'])
# print(result["parsed"].imports + "\n" + result["parsed"].code)


class TestCase(TypedDict):
    inputs: str
    outputs: str


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        code_solution : Code solution
        iterations : Number of tries
        test_cases : List of test cases
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    code_solution: str
    iterations: int
    test_cases: list[TestCase]


# Parameters
max_iterations = 10

# Nodes


def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, code_solution
    """

    print("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # Solution
    code_solution = code_gen_chain.invoke({"messages": messages})['parsed']
    if os.getenv("DEBUG_MODE") == "True":
        print(code_solution)

    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"code_solution": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    print("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["code_solution"]
    iterations = state["iterations"]
    test_cases = state["test_cases"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        print("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            ("user", f"Your solution failed the import test. Here is the error: {e}. Reflect on these errors and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:")]
        messages += error_message
        return {
            "code_solution": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }
    # Check execution
    combined_code = f"{imports}\n{code}"
    # Use a shared scope for exec
    num_test_cases = len(test_cases)
    succeeded = 0
    test_results = []
    for test_case in test_cases:
        global_scope = {
            "global_input": test_case["inputs"],
            "global_output": ""
        }
        try:
            exec(combined_code, global_scope)
            if str(global_scope["global_output"]) == str(test_case["outputs"]):
                test_results.append(f"Correct test case! Your output matched the expected output: {test_case['outputs']}")
                succeeded += 1
            else:
                test_results.append(f"Wrong answer. Got {global_scope['global_output']} but expected {test_case['outputs']}")
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            test_results.append(f"Wrong answer. Got exception {e} but expected {test_case['outputs']}")
        pass_rate = succeeded / num_test_cases if num_test_cases else "N/A"
    if succeeded == num_test_cases:
        print("---CODE BLOCK CHECK: SUCCEEDED---")
        return {
            "code_solution": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        }
    print("---CODE BLOCK CHECK: FAILED---")
    responses = "\n".join([f"<test id={i}>\n{r}\n</test>" for i, r in enumerate(test_results)])
    response = f"Incorrect submission.\nPass rate: {succeeded}/{num_test_cases}\nResults:\n{responses}"
    error_message = [
        ("user", f"{response} Reflect on these test case failures and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:")]
    messages += error_message
    return {
        "code_solution": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "yes",
    }


# Conditional edges
def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"

# Utilities


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print(f"Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("generate", generate)  # code_solution solution
builder.add_node("check_code", code_check)  # check code

# Build graph
builder.set_entry_point("generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

memory = SqliteSaver.from_conn_string(":memory:")
graph = builder.compile(checkpointer=memory)

try:
    graph_image = graph.get_graph().draw_mermaid_png()
    with open("data/graph.png", "wb") as f:
        f.write(graph_image)
    print("Graph saved as 'data/graph.png'.")
except Exception as e:
    print(f"Failed to save graph: {e}")

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

# Run it
import utils

response_content1 = utils.load_response_data('data/response.txt')
response_content2 = utils.load_response_data('data/response2.txt')

# Find a few more interesting cases.
# If they fail or take many iterations, try with a debug node.
USER_REQUEST = f"""Write a function to extract and list all available playing times 
             from HTTP response content.
             
             Example HTTP response content:
               {response_content1}

            If there are multiple activity types, use a dictionary instead of a list and associate times
            with the correct activity.            

            Please pay attention to the system message above to see how to format your response.
            """

TEST_CASES = [
            {"inputs": f"{response_content1}", "outputs": "{'Pickleball / Mini Tennis': ['9:30pm'], 'Tennis': ['9:30pm']}"},
            {"inputs": f"{response_content2}", "outputs": "{'Pickleball / Mini Tennis': ['9:00pm', '9:30pm'], 'Tennis': ['8:00pm', '8:30pm', '9:00pm', '9:30pm']}"},
        ]

events = graph.stream(
    {"messages": [("user", USER_REQUEST)], "iterations": 0, "test_cases": TEST_CASES}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

print(event['code_solution'])
