from langchain import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from collections import OrderedDict
from PIL import Image
import google.generativeai as genai
# import vertexai
# from openai import OpenAI
# from IPython.display import Image, display, Audio, Markdown
import pandas as pd
import re
import numpy as np
import io
import base64
from langchain_core.messages import HumanMessage

MODEL="gpt-4"

import tempfile




generation_config = genai.GenerationConfig(temperature=0.7)
# generation_config = GenerationConfig(temperature=0)
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# Set up the LLM model (OpenAI GPT in this case)

llm = ChatOpenAI(model_name=MODEL)

# Set up memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history")

# Define prompt template for task decomposition

template = """
You are a robot with wheels and a mechanical arm that can move up and down, stretch out and draw back. On the end of the arm, there is a gripper that is in open state by default. You will be given a task and a list of available skill functions.
You should complete the given task by only using these skill functions to compose a new function as a new skill.
To make the new skill more universal, you should carefully name it and structure it to make it useable to a class of tasks that are similar to the given task.
Finally, you should also call your new skill function with proper setting. 
Please start with a reasoning step and then write your code in triplets.


Your task is: {task}

Following are the sensing skills:
1. locate(object_name): return the location of the object in bouding box with format [xmin, ymin, xmax, ymax]
2. observe(object_name, subtask:str): write a description of the object for guidance of how to complete the subtask into the memory, return nothing

Following are the acting skills:
1. move(x, y): move gripper to a location x, y
2. close_gripper(): close the gripper
3. open_gripper(): open the gripper
4. grasp(object_name): locate and grasp the object, return the cropped image for the object
5. place(object_name, cropped_image): locate the recepter object 'object_name' and place the things grasped by the gripper to the recepter object, return a final state image
6. move_gripper(x, y, z, theta):move gripper to a distance x, y, z and rotate an angle theta

Following are controlling skills:
1. while_loop([skills], condition): repeat a list of skills while condition is true
2. repeat([skills], times): repeat a list of skills in several times
3. stop(): Stops all movement of the robot when all tasks are completed.
4. wait(condition, time): Given condition and time, robot will wait for a maximum seconds of time or wait until the condition is met before next skill execution

{chat_history}
"""

prompt = PromptTemplate(input_variables=["task", "chat_history"], template=template)

IMG = Image.open("example/water_dispenser.jpg")

buffered = io.BytesIO()
IMG.save(buffered, format="PNG")
img_bytes = buffered.getvalue()
base64_encoded = base64.b64encode(img_bytes).decode('utf-8')


from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-exp-0827", api_key=google_api_key)
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)