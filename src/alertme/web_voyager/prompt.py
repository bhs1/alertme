from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.messages.tool import ToolMessage

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate
)
from langchain.schema import AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage
from typing import List, Union

# System message template
task_param_used_prompt = """Imagine you were a robot browsing the web, just like humans. You needed to complete a task. \
In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will \
feature Numerical Labels placed in the BOTTOM RIGHT corner of each Web Element. Carefully analyze the visual \
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow \
the guidelines and choose ONE of the following actions:

1. Click: Click a Web Element.
2. Type: Delete existing text in a textbox, type text, then press enter.
3. Scroll: Scroll up or down until text is visible. ATTENTION: The default scroll is the whole window. \
But if the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element \
in that area using Numerical_Label.
- ATTENTION: If Scroll is not an available action for an element, then you should instead scroll the nearest scrollable element in the image.
4. Wait
5. Go back
7. Return to google to start over.
8. Go to a specific URL
9. Respond with the final answer.

Correspondingly, raw_predicted_action_string should STRICTLY follow the format:
- Click [Numerical_Label] E.g. Click [30]
- Type [Numerical_Label], [Text] E.g. Type [30, "Some Text"]
- Scroll [Numerical_Label or WINDOW], [up or down], [Text] e.g. Scroll [30, down, "Some Text we want to find."]
- Wait 
- GoBack
- Google
- ToUrl [Text] E.g. ToUrl www.mywebsite.com
- ANSWER [Content]

Imagine, that you have ALREADY chosen the action to take: {raw_predicted_action_string}

This was your thought for why to take this action: {action_thought}

The text associated with the bounding box you interacted with is: {bbox_text}

Your task now: we would like to understand which key value pair, if any, from this task_params map was used to generate the [Text] in the action string.

task_params map: {task_params}

You should choose task_param_used as follows:
- If you are using [Text] derived from task_params map below, output the name of the task_param in task_param_used field.
- Double check task_params values for [Text], if [Text] is a value in the task_param map, then it's likely its key should be in task_param_used.
- But it could also be a derived value. E.g. if [Text] is "100 dollars", then it's likely the task_param used is "price" with value 100.
- It is also possible that no task_params are used to generate [Text]. In that case, output "None" for task_param_used.

Your reply should strictly follow the format:

task_param_used_explanation: {{Your explanation of task_param_used}} E.g. "No task_params are needed for this action because..." or "chose task_param date because it was used to determine [Text]."
task_param_used: {{Name of the task_param used to fill the [Text] in the action string. Be sure it is a key in task_params map above.}}

Your overall objective was:

{task}

Here are the actions that were available to you:

{bbox_descriptions}

And below is a screenshot of the webpage you are currently on:
"""

# System message template
system_template = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. \
In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will \
feature Numerical Labels placed in the BOTTOM RIGHT corner of each Web Element. Carefully analyze the visual \
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow \
the guidelines and choose ONE of the following actions:

1. Click: Click a Web Element.
2. Type: Delete existing text in a textbox, type text, then press enter.
3. Scroll: Scroll up or down until text is visible. ATTENTION: The default scroll is the whole window. \
But if the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element \
in that area using Numerical_Label.
- ATTENTION: If Scroll is not an available action for an element, then you should instead scroll the nearest scrollable element in the image.
4. Wait
5. Go back
7. Return to google to start over.
8. Go to a specific URL
9. Respond with the final answer.

Correspondingly, raw_predicted_action_string should STRICTLY follow the format:
- Click [Numerical_Label] E.g. Click [30]
- Type [Numerical_Label], [Text] E.g. Type [30, "Some Text"]
- Scroll [Numerical_Label or WINDOW], [up or down], [Text] e.g. Scroll [30, down, "Some Text we want to find."]
- Wait 
- GoBack
- Google
- ToUrl [Text] E.g. ToUrl www.mywebsite.com
- ANSWER [Content]

Key Guidelines You MUST follow:
* Action guidelines *
0) For actions with Numerical_Label, you should only take the action if the Numerical_Label is available in the "Available actions" field.
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the bottom-right corner of their corresponding bounding boxes and are colored the same.
4) Don't try the same action multiple times if it keeps failing.
5) When trying to understand text in the images, check the "Text" field in "Available actions" for the answer.

Your reply should strictly follow the format:
Thought: {{Your brief thoughts}}
Action: {{One Action format you choose}}
Action args: {{List of action args}}
"""

human_task_param_used_prompt = HumanMessagePromptTemplate.from_template(task_param_used_prompt)

image_prompt = ImagePromptTemplate(
    input_variables=["img"],
    template={"url": "data:image/png;base64,{img}"}
)

human_current_screenshot_prompt = HumanMessagePromptTemplate(prompt=[image_prompt])

task_param_used_prompt_template = ChatPromptTemplate(
    messages=[
        human_task_param_used_prompt,
        human_current_screenshot_prompt
    ]
)

param_conversion_code_prompt_template = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template(
"""
Generate code to convert this input string: {input_string} to this output string: {output_string}. The code should be a python function of the form:

def convert(input_string):
    # Add code to extract output_string from input_string>    
    return output_string
"""
    )
])

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

bbox_prompt = PromptTemplate(
    input_variables=["bbox_descriptions"],
    template="{bbox_descriptions}"
)

input_prompt = PromptTemplate(
    input_variables=["task"],
    template="""

Overall goal:
{task}

"""
)

params_prompt = PromptTemplate(
    input_variables=["task_params"],
    template="""

task_params:
{task_params}

"""
)

scratchpad_prompt = PromptTemplate(
    input_variables=["formatted_scratchpad"],
    template="""

Most recent actions:
{formatted_scratchpad}

"""
)

human_message_prompt = HumanMessagePromptTemplate(prompt=[scratchpad_prompt, image_prompt, bbox_prompt, input_prompt, params_prompt])

# Chat prompt template
web_voyager_prompt = ChatPromptTemplate(
    input_variables=["bbox_descriptions", "img", "task", "formatted_scratchpad"],
    messages=[
        system_message_prompt,
        human_message_prompt
    ]
)

# Image prompt for new screenshot
new_screenshot_prompt = ImagePromptTemplate(
    input_variables=["new_screenshot"],
    template={"url": "data:image/png;base64,{new_screenshot}"}
)

# Image prompt for input annotated screenshot
old_screenshot_prompt = ImagePromptTemplate(
    input_variables=["old_screenshot"],
    template={"url": "data:image/png;base64,{old_screenshot}"}
)

final_screenshot_prompt = ImagePromptTemplate(
    input_variables=["final_screenshot"],
    template={"url": "data:image/png;base64,{final_screenshot}"}
)

human_extract_info_prompt = HumanMessagePromptTemplate.from_template(
"""
According to this prompt
    
{input_prompt}

{params_prompt}

extract the information you need from the screenshot below.

Be sure to follow this output format example when you extract the result:

{output_format_example}
"""
)

concatenated_extract_info_prompt = HumanMessagePromptTemplate(prompt=[human_extract_info_prompt, final_screenshot_prompt])

final_extract_info_prompt = ChatPromptTemplate(
    input_variables=["final_screenshot", "input_prompt", "output_format_example"],
    messages=[
        concatenated_extract_info_prompt
    ]
)


# Action taken prompt
action_taken_prompt = PromptTemplate(
    input_variables=["action_taken", "task"],
    template="""Action taken: {action_taken}
    """
)

reflection_human_prompt = HumanMessagePromptTemplate(prompt=[old_screenshot_prompt, action_taken_prompt, new_screenshot_prompt])

# Combined prompt
# TODO: Add the overall prompt message as well and say not to forget it (this should fix the 9am 9pm messup).
compare_screenshots_prompt = ChatPromptTemplate(
    input_variables=["new_screenshot", "action_taken", "input_annotated_screenshot"],
    messages=[
        reflection_human_prompt,
        SystemMessagePromptTemplate.from_template("""
- Summarize the intention of the action.

- Summarize the result of the action by comparing the before (first) and after (second) screenshots. Be very careful \
here to check the details of the second screenshot. Don't make stuff up.

- Explain why the result is one of:
  - SUCCESS: Action result matched the intent.
  - FAILURE: Action result did not match the intent.
  - PARTIAL_SUCCESS: Action result did not match the intent but did make partial progress towards the "Thought".
  
Print out SUCCESS, FAILURE or PARTIAL_SUCCESS.
    """),
    ]
)