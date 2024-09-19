# import sys
# sys.path.insert(0, '.')
# sys.path.insert(0, './web_voyager')
#
# from langchain_openai import ChatOpenAI
# from langchain_core.pydantic_v1 import BaseModel, Field
# from typing import List
# from dotenv import load_dotenv
# import os
# import json
# from web_voyager.replay_actions import ActionReplayer
# from codegen import exec_as_main
# from langsmith import traceable
# from web_voyager.utils import compress_screenshot
#
# # Apply code to actions_log
# # Replay new actions_log
#
# class LLMOutput(BaseModel):
#     postprocessed_output: str = Field(
#         description="The postprocessed output of the task in the prompt param, formatted correctly.")
#
# from web_voyager.prompt import final_extract_info_prompt
# llm_chain = final_extract_info_prompt | ChatOpenAI(model_name="gpt-4o", temperature=0.0).with_structured_output(LLMOutput)
#
# #TODO(0): Figure out how to close windows, then watch and figure out why times are not being correctly replaced.
# @traceable(name="generalize_actions")
# async def execute_code(code, global_scope, additional_params = {}):
#     try:
#         exec_as_main(code, global_scope)
#     except Exception as e:
#         return {"error" : f"Exception: {e}", "postprocessed_output" : ""}
#
#     if global_scope["global_output"] == "":
#         return {"error" : "No output from code execution.", "postprocessed_output" : ""}
#
#     updated_action_log = global_scope["global_output"]
#
#     try:
#         replayer = ActionReplayer(actions_string=updated_action_log)
#         await replayer.replay_actions()
#     except Exception as e:
#         return {"error" : f"Exception while replaying actions: {e}. Please check if actions are well formed."}
#
#     import asyncio
#
#     page = await replayer.get_page()
#     # Take a screenshot of the page
#     screenshot = await page.screenshot()
#     screenshot = await compress_screenshot(screenshot)
#     print("Calling LLM to extract final output...")
#     start_time = asyncio.get_event_loop().time()
#     # TODO(0) fix errors here.
#     # TODO: Add replay here, also add scroll to value ability if nothing visible?
#     out = llm_chain.invoke({"final_screenshot": screenshot, **additional_params})
#     end_time = asyncio.get_event_loop().time()
#     print(f"Done. Time taken: {end_time - start_time:.2f} seconds")
#     postprocessed_output = out.postprocessed_output
#     # Ask LLM whether the task in the prompt param (we need to pass prompt param) has been completed
#     # successfully and if it has, ask the LLM to extract information from the page of the form of the
#     # exepcted output (pass as param). Use with_structured_output. If not, return an explanation of what is different between the
#     # expected output and the actual output. Try to format the actual output like the expected output and return that as postprocessed_output string.
#     # Return whether there was a success, and what was different.
#     await page.close()
#     return {"postprocessed_output" : postprocessed_output}
#
# def get_prompt(test_cases):
#     os.environ["LANGCHAIN_PROJECT"] = "generalizer"
#     # Define the prompt for the LLM
#     prompt = f"""Write a python function func that takes a dictionary argument. The dictionary will have two arguments
#     "params" and "action_log". The function should return a new action log (str) which is updated according to params.
#
#     The value for key "params" is of the form:
#
#     {test_cases[0]["inputs"]["params"]}
#
#     The value for key "action_log" is of the form:
#
#     {test_cases[0]["inputs"]["action_log"]}
#
#     The result of the function should be a valid json of the same form as action_log. We will run the result of your function to produce
#     the final output to be compared against the test cases.
#     """
#     return prompt