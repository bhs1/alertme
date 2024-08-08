import sys

sys.path.insert(0, '.')
sys.path.insert(0, './web_voyager')

from dotenv import load_dotenv
from web_voyager.playwright_actions import setup_browser
from web_voyager.prompt import compare_screenshots_prompt
import io
from PIL import Image
import json
from web_voyager.playwright_actions import click as playwright_click, type_text as playwright_type_text, scroll as playwright_scroll, go_back as playwright_go_back, to_google as playwright_to_google, to_url as playwright_to_url
import logging
from web_voyager.prompt import web_voyager_prompt
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from web_voyager.utils import mask_sensitive_data
from langsmith import traceable
from langgraph_utils import generate_graph_image
import os
import uuid

from typing import List, Optional, TypedDict

from playwright.async_api import Page

import asyncio

import base64

from langchain_core.runnables import chain as chain_decorator
from langchain_core.tracers.langchain import wait_for_all_tracers

from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# TODO: Make this file object oriented instead of using a global page object.
global page

# Ensure the logs directory exists
log_dir = "./web_voyager/logs"
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=os.path.join(log_dir, 'info.txt'),
                    filemode='a')

os.environ["LANGCHAIN_PROJECT"] = "Web-Voyager-3"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "true"

class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str
    outerHTML: str
    parentHTML: str
    isScrollable: bool
    isTypeable: bool
    isClickable: bool
    index: str
    xpath: str


class Prediction(BaseModel):
    thought: Optional[str] = Field(description="Thought: Your brief thoughts.")
    raw_predicted_action_string: Optional[str] = Field(
        description="Prediction action string and args, formatted as described.")
    action: str = Field(description="One action you choose")
    args: Optional[List[str]] = Field(description="List of action args.")


class Reflection(BaseModel):
    """Reflection output"""
    intention_of_action: str = Field(
        description="Intention of the attempted action.")
    result_of_action: str = Field(
        description="Description of relevant details before screenshot, after screenshot, and comparison of the two.")
    explanation: str = Field(
        description="Brief explanation of why the action was a success, partial success, or failure.")
    success_or_failure_result: str = Field(
        description="Must be one of SUCCESS, PARTIAL_SUCCESS, or FAILURE.")


async def compress_screenshot(screenshot, scale_factor: int = 1) -> str:
    """Compress the screenshot by reducing its width and height by 4x."""
    image = Image.open(io.BytesIO(screenshot))

    new_size = (image.width // scale_factor, image.height // scale_factor)
    compressed_image = image.resize(new_size, Image.LANCZOS)

    # Convert back to base64
    buffer = io.BytesIO()
    # Save as PNG to avoid JPEG compression
    compressed_image.save(buffer, format="PNG")
    compressed_b64_image = base64.b64encode(buffer.getvalue()).decode()

    return compressed_b64_image

# This represents the state of the agent
# as it proceeds through execution


class AgentState(TypedDict):
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[str]
    observation: str
    action_summary: str
    step: int
    formatted_scratchpad: str


actions_log = []


def store_action(action_name: str, params: dict):
    actions_log.append({"action": action_name, "params": params})


async def click(state: AgentState):
    try:
        global page
        click_args = state["prediction"].args
        if click_args is None or len(click_args) != 1:
            return f"Failed to click bounding box labeled as number {click_args}"
        bbox_id = int(click_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        xpath = bbox.get("xpath", "")
        click_result = await playwright_click(page, x, y, bbox.get("text", ""))
        print(f"Clicked {bbox_id} with result {click_result}")        
        store_action("click", {"x": x, "y": y,
                     "bbox_id": bbox_id, "xpath": xpath, "click_result": click_result, "text": bbox.get("text", ""), "parent_html": bbox.get("parentHTML", "")})
        return f"Clicked {bbox_id}"
    except Exception as e:
        return f"Error in click: {str(e)}"


async def type_text(state: AgentState):
    try:
        global page
        type_args = state["prediction"].args
        if type_args is None or len(type_args) != 2:
            return f"Failed to type in element from bounding box labeled as number {type_args}"
        bbox_id = int(type_args[0])
        bbox = state["bboxes"][bbox_id]
        x, y = bbox["x"], bbox["y"]
        text_content = type_args[1]
        xpath = bbox.get("xpath", "")
        await playwright_type_text(page, x, y, text_content)
        store_action("type_text", {
                     "x": x, "y": y, "bbox_id": bbox_id, "text": text_content, "xpath": xpath})
        return f"Typed {text_content} and submitted"
    except Exception as e:
        return f"Error in type_text: {str(e)}"

async def scroll(state: AgentState):
    try:
        global page
        scroll_args = state["prediction"].args
        if scroll_args is None or len(scroll_args) != 2:
            return "Failed to scroll due to incorrect arguments."

        target, direction = scroll_args

        if target.upper() == "WINDOW":
            x, y = 0, 0
            xpath = ""
            scroll_amount = 500
        else:
            x, y = state["bboxes"][int(
                target)]["x"], state["bboxes"][int(target)]["y"]
            xpath = state["bboxes"][int(target)].get("xpath", "")
            scroll_amount = 200
        scroll_direction = -scroll_amount if direction.lower() == "up" else scroll_amount
        await playwright_scroll(page, x, y, direction, scroll_direction)
        store_action("scroll", {"x": x, "y": y, "direction": direction,
                     "scroll_direction": scroll_direction, "xpath": xpath})
        return f"Scrolled {direction} in {'window' if target.upper() == 'WINDOW' else 'element'}"
    except Exception as e:
        return f"Error in scroll: {str(e)}"


async def wait(state: AgentState):
    try:
        sleep_time = 5
        await asyncio.sleep(sleep_time)
        store_action("wait", {"sleep_time": sleep_time})
        return f"Waited for {sleep_time}s."
    except Exception as e:
        return f"Error in wait: {str(e)}"


async def go_back(state: AgentState):
    try:
        global page
        await playwright_go_back(page)
        store_action("go_back", {})
        return f"Navigated back a page to {page.url}."
    except Exception as e:
        return f"Error in go_back: {str(e)}"


async def to_google(state: AgentState):
    try:
        global page
        await playwright_to_google(page)
        store_action("to_google", {})
        return "Navigated to google.com."
    except Exception as e:
        return f"Error in to_google: {str(e)}"


async def to_url(state: AgentState):
    try:
        global page
        url = state["prediction"].args[0]
        await playwright_to_url(page, url)
        store_action("to_url", {"url": url})
        return f"Navigated to {url}."
    except Exception as e:
        return f"Error in to_url: {str(e)}"

# Some javascript we will run on each step
# to take a screenshot of the page, select the
# elements to annotate, and add bounding boxes
with open("./web_voyager/mark_page.js") as f:
    mark_page_script = f.read()


@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            # May be loading...
            asyncio.sleep(3)
    screenshot = await take_screenshot(page)
    # Ensure the bboxes don't follow us around
    await page.evaluate("unmarkPage()")
    return {
        "img": screenshot,
        "bboxes": bboxes,
    }


async def annotate(state):    
    marked_page = await mark_page.with_retry().ainvoke(page)
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for bbox in state["bboxes"]:
        text = bbox.get("text", "").strip()
        scrollable = bbox.get("isScrollable", False)
        typable = bbox.get("isTypeable", False)
        clickable = bbox.get("isClickable", False)
        index = bbox.get("index", "")

        available_actions = []
        if scrollable:
            available_actions.append(f"Scroll [{index}]")
        if clickable:
            available_actions.append(f"Click [{index}]")
        if typable:
            available_actions.append(f"Type [{index}]")

        actions_str = ", ".join(available_actions)        

        labels.append(f'''(Box {index}) {actions_str}
    - Text: "{text}"
''')
    bbox_descriptions = "\nAvailable actions:\n" + "\n".join(labels)
    print("Calling LLM to determine action...")
    return {**state, "bbox_descriptions": bbox_descriptions}


# Chains
llm = ChatOpenAI(model="gpt-4o", temperature=0)

agent = annotate | RunnablePassthrough.assign(
    prediction=format_descriptions | web_voyager_prompt | llm.with_structured_output(Prediction)
)

compare_chain = compare_screenshots_prompt | llm.with_structured_output(Reflection)

async def setup(state: AgentState):
    global page
    #task = f"""Book me a tennis court at mission dolores park tomorrow."""
    #task = f"""Go to https://bot.sannysoft.com/ and see which checks we fail"""
    #task = f"""Go to https://nowsecure.nl and see which checks we fail"""
    #task = """Go to google and then google hello and then scroll to the bottom of the window."""
    username = os.getenv('GTC_USERNAME')
    password = os.getenv('GTC_PASSWORD')
    task = f"""Go to url: https://gtc.clubautomation.com/. Use username: {username}, password: {password}.
Task: Search for courts that satisfy the criteria (do not reserve):
 - date: 08/12/2024
 - from_time: 10am
 - to_time: 9pm
 - duration: 60 mins.
 
 Print the available times.
"""
    page = await setup_browser()    
    return {**state, "input": task}

async def update_scratchpad(state: AgentState):
    """After a tool is invoked, we want to update
    the scratchpad so the agent is aware of its previous steps"""
    old = state.get("scratchpad", list())
    step = state.get("step")
    if step is None:
        step = 1

    args = state['prediction'].args or []
    bbox_id = None
    if args:
        try:
            bbox_id = int(args[0])
        except ValueError:
            bbox_id = None

    modified_args = args.copy()
    if bbox_id is not None:
        modified_args[0] = "Numerical_Label"

    bbox_text = state['bboxes'][bbox_id]['text'] if bbox_id is not None else "N/A"
    action_attempted = f"Action attempted: {state['prediction'].action} with args: {modified_args}"
    if bbox_text and bbox_text != "N/A":
        action_attempted += f", bounding box text: '{bbox_text}'"

    latest_step = f"""{action_attempted}

{state['action_summary']}
    """
    # print("================== STEP ====================")
    # print(latest_step)
    old.append(latest_step)
    if len(old) > 3:
        old = old[-3:]  # Keep only the latest 3 steps

    step += 1
    formatted_scratchpad = format_scratchpad(old)

    return {**state, "scratchpad": old, "formatted_scratchpad": formatted_scratchpad, "step": step}


def format_scratchpad(scratchpad):
    content = "\n----------------------------------\n".join(
        [f"{i+1}. {entry}" for i, entry in enumerate(scratchpad)])
    return "======================================================\n" + content + "\n======================================================"


async def take_screenshot(page):
    screenshot = await page.screenshot()
    compressed_screenshot = await compress_screenshot(screenshot)
    return compressed_screenshot

@traceable(name="reflect")
async def reflect(state: AgentState):
    """
    (1) Take a screenshot of the page
    (2) Compare it to the previous screenshot before the action
    (3) Summarize the result of the action and whether it succeeded or failed
    (4) Append any error to the summary
    (5) Return the summary in the agent state
    """

    try:
        bbox_id = int(state['prediction'].args[0]
                      ) if state['prediction'].args else None
    except ValueError:
        bbox_id = None
    bbox_text = state['bboxes'][bbox_id]['text'] if bbox_id is not None else "N/A"
    if bbox_text and bbox_text != "N/A":
        print_str = f"\033[1mExecuted action: {state['prediction'].action} with args: {state['prediction'].args}, Bounding box text: {bbox_text}\033[0m"
    else:
        print_str = f"\033[1mExecuted action: {state['prediction'].action} with args: {state['prediction'].args}\033[0m"
    masked_print_str = await mask_sensitive_data(print_str)
    
    print(masked_print_str)

    global page
    prediction = state.get("prediction")
    action_taken = f"Action attempted: {prediction.action} with args: {prediction.args}. Which results in:"
    thought = prediction.thought
    if thought:
        action_taken = f"Thought: {thought}\n" + action_taken
    old_screenshot = state.get("img")
    # new_screenshot = await take_screenshot(page)  # Take a new screenshot of the page
    # Call annotate on the new screenshot
    annotated_state = await annotate({"page": page})
    new_screenshot = annotated_state["img"]

    # Save the images to files in the scratch directory
    def save_image_to_file(image_data, filename):
        file_path = os.path.join(
            "./web_voyager/scratch", filename)
        with open(file_path, "wb") as file:
            file.write(image_data.encode('utf-8'))

    save_image_to_file(new_screenshot, "new_screenshot.jpg")
    save_image_to_file(old_screenshot, "old_screenshot.jpg")
    run_id = str(uuid.uuid4())
    print("Calling LLM to reflect on action...")
    start_time = asyncio.get_event_loop().time()
    comparison_result = compare_chain.invoke({"new_screenshot": new_screenshot, "action_taken": action_taken, "old_screenshot": old_screenshot,
                                   "input": state["input"]}, {"run_name": "ReflectOnDiff", "tags": ["reflect-on-diff", run_id]})
    end_time = asyncio.get_event_loop().time()
    print(f"Time taken for comparison: {end_time - start_time:.2f} seconds")
    
    start_time = asyncio.get_event_loop().time()
    print("Waiting for traces to upload...")
    wait_for_all_tracers()
    end_time = asyncio.get_event_loop().time()
    print(f"Traces uploaded. Time taken: {end_time - start_time:.2f} seconds")

    explanation = comparison_result.explanation
    success_or_failure_result = comparison_result.success_or_failure_result
    summary = f"""Action result:
        {success_or_failure_result}
Explanation:
    {explanation}
"""
    if "FAILURE" in summary:
        print(f"\033[1;31m{summary}\033[0m")  # Bold red
    else:
        print(f"\033[1;32m{summary}\033[0m")  # Bold green
    print(f"Run ID: {run_id}")

    if "error" in state:
        summary += f"\nError: {state['error']}"

    state["action_summary"] = summary
    return state

graph_builder = StateGraph(AgentState)


graph_builder.add_node("agent", agent)

tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
    "ToUrl": to_url,
}

for node_name, tool in tools.items():
    graph_builder.add_node(
        node_name,
        RunnableLambda(tool) | (lambda observation: {
            "observation": observation}),
    )


def select_tool(state: AgentState):
    # Any time the agent completes, this function
    # is called to route the output to a tool or
    # to the end user.
    action = state["prediction"].action
    if "ANSWER" in action:
        return "finish"
    return action

async def end(state: AgentState):
    res = state["prediction"].args[0] # TODO: Format this.
    masked_res = await mask_sensitive_data(res)
    print(f"Final response: {masked_res}")

    await save_actions_log()
    wait_for_all_tracers()
    
graph_builder.add_conditional_edges("agent", select_tool)
graph_builder.add_node("reflect", RunnableLambda(reflect))

for node_name, tool in tools.items():
    graph_builder.add_edge(node_name, "reflect")

graph_builder.add_node("update_scratchpad", RunnableLambda(update_scratchpad))
graph_builder.add_edge("reflect", "update_scratchpad")
graph_builder.add_edge("update_scratchpad", "agent")

graph_builder.add_node("setup", setup)
graph_builder.add_node("finish", end)
graph_builder.add_edge("finish", END)
graph_builder.set_entry_point("setup")
graph_builder.add_edge("setup", "agent")

graph = graph_builder.compile()
generate_graph_image(
    graph, filename="./web_voyager/logs/web_voyager_graph.png")


async def call_agent(max_steps: int = 300):
    event_stream = await graph.ainvoke(
        {
            "scratchpad": list(),
        },
        {
            "recursion_limit": max_steps,
        },
    )

# Load environment variables from .env file
load_dotenv()

async def save_actions_log():
    log_path = "./web_voyager/data/actions_log.json"
    with open(log_path, "w") as f:
        json.dump(actions_log, f, indent=4)


if __name__ == "__main__":
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(call_agent())
    finally:
        loop.close()
        sys.stderr = original_stderr
