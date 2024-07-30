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
system_template = """Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will
feature Numerical Labels placed in the BOTTOM RIGHT corner of each Web Element. Carefully analyze the visual
information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow
the guidelines and choose ONE of the following actions:

1. Click: Click a Web Element.
2. Type: Delete existing content in a textbox, type content, then press enter.
3. Scroll: Scroll up or down. Multiple scrolls are allowed to browse the webpage. ATTENTION: The default scroll is the whole window. But if the scroll widget is located in a certain area of the webpage, then you have to specify a Web Element in that area using Numerical_Label.
 - ATTENTION: If Scroll is not an available action for an element, then you should instead scroll the nearest scrollable element in the image.
4. Wait
5. Go back
7. Return to google to start over.
8. Go to a specific URL
9. Respond with the final answer

Correspondingly, raw_predicted_action_string should STRICTLY follow the format:

- Click [Numerical_Label];
- Type [Numerical_Label]; [Content] 
- Scroll [Numerical_Label or WINDOW]; [up or down] e.g. Scroll [30]; down 
- Wait 
- GoBack
- Google
- ToUrl [URL]
- ANSWER; [content]

Key Guidelines You MUST follow:

* Action guidelines *
0) For actions with Numerical_Label, you should only take the action if the Numerical_Label is available in the "Available actions" field.
1) Execute only one action per iteration.
2) When clicking or typing, ensure to select the correct bounding box.
3) Numeric labels lie in the bottom-right corner of their corresponding bounding boxes and are colored the same.
4) Don't try the same action multiple times if it keeps failing.
5) When trying to understand text in the images, check the "Text" field in "Available actions" for the answer.

* Web Browsing Guidelines *
1) Don't interact with useless web elements like Login, Sign-in, donation that appear in Webpages
2) Select strategically to minimize time wasted.

Your reply should strictly follow the format:

Thought: {{Your brief thoughts}}
Action: {{One Action format you choose}}
Action args: {{List of action args}}
Then the User will provide:
Observation: {{A labeled screenshot Given by User}}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

# Human message template
image_prompt = ImagePromptTemplate(
    input_variables=["img"],
    template={"url": "data:image/png;base64,{img}"}
)

bbox_prompt = PromptTemplate(
    input_variables=["bbox_descriptions"],
    template="{bbox_descriptions}"
)

input_prompt = PromptTemplate(
    input_variables=["input"],
    template="""

Overall goal:
{input}

"""
)

scratchpad_prompt = PromptTemplate(
    input_variables=["formatted_scratchpad"],
    template="""

Most recent actions:
{formatted_scratchpad}

"""
)

human_message_prompt = HumanMessagePromptTemplate(prompt=[image_prompt, bbox_prompt, input_prompt, scratchpad_prompt])

# Chat prompt template
web_voyager_prompt = ChatPromptTemplate(
    input_variables=["bbox_descriptions", "img", "input", "formatted_scratchpad"],
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

# Action taken prompt
action_taken_prompt = PromptTemplate(
    input_variables=["action_taken", "input"],
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