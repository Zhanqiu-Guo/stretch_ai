import inspect
import ast
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
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
import glob
# import vertexai
# from openai import OpenAI
# from IPython.display import Image, display, Audio, Markdown

import pandas as pd
import re
import numpy as np
import io
import base64
from langchain_core.messages import HumanMessage

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
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""Based on the following conversation history:
{chat_history}
Human: {input}
Assistant:"""
)

def compute_tilt(camera_xyz, target_xyz):
    """
    a util function for computing robot head tilts so the robot can look at the target object after navigation
    - camera_xyz: estimated (x, y, z) coordinates of camera
    - target_xyz: estimated (x, y, z) coordinates of the target object
    """
    if not isinstance(camera_xyz, np.ndarray):
        camera_xyz = np.array(camera_xyz)
    if not isinstance(target_xyz, np.ndarray):
        target_xyz = np.array(target_xyz)
    vector = camera_xyz - target_xyz
    return -np.arctan2(vector[2], np.linalg.norm(vector[:2]))

memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-exp-0827", api_key=google_api_key)
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


class RobotCodeGen:
    def __init__(self, robot, demo):
        self.robot = robot
        self.demo = demo
        self.state = {}
        self.history = ''
        self.object_on_hand = 'nothing'
        self.task_planned = []
        self.task_finished = []
        self.task_decomposed = ''
        self.safe_globals = {
            'navigate': self.navigate,
            'pick_up': self.pick_up,
            'place_to': self.place_to,
            'press': self.press,
            'robot': self.robot, 
            'demo': self.demo
        }

    def navigate(self, object_name):
        print(f'Navigating to {object_name}')
        self.robot.move_to_nav_posture()
        self.robot.switch_to_navigation_mode()
        point = self.demo.navigate(object_name)
        if point is None:
            print("Navigation Failure!")
        self.robot.switch_to_navigation_mode()
        xyt = self.robot.get_base_pose()
        xyt[2] = xyt[2] + np.pi / 2
        self.robot.move_base_to(xyt, blocking=True)
        self.state['point'] = point
        return True

    def pick_up(self, object_name):
        print(f'Picking up {object_name}')
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        point = self.state['point']
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6
        self.demo.manipulate(object_name, theta)
        self.robot.look_front()
        self.object_on_hand = object_name
        return True

    def press(self, object_name):
        print(f'Pressing {object_name}')
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        point = self.state['point']
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6
        self.demo.press(object_name, theta)
        self.robot.look_front()
        return True
    
    def place_to(self, object_name):
        print(f'Placing at {object_name}')
        self.robot.switch_to_manipulation_mode()
        camera_xyz = self.robot.get_head_pose()[:3, 3]
        point = self.state['point']
        if point is not None:
            theta = compute_tilt(camera_xyz, point)
        else:
            theta = -0.6
        self.demo.place(object_name, theta)
        self.robot.move_to_nav_posture()
        self.object_on_hand = 'nothing'
        return True

    # def sub_task_complete(self, task: str) -> None:        
    #     genai.configure(api_key=google_api_key_2)
    #     model = genai.GenerativeModel(model_name=f"models/gemini-1.5-flash-exp-0827")


    #     user_prompt = f"""
    #         You are a robot with wheels and a mechanical arm that can move up and down, stretch out and draw back. 
    #         On the end of the arm, there is a gripper that is in open state by default, it could act like a rod when it is closed. 
    #         You will be given a task and a list of available skill functions.
    #         You should complete the given task by only using these skill functions to compose a new function as a new skill.
    #         To make the new skill more universal, you should carefully name it and structure it to make it useable to a class of tasks that are similar to the given task.
    #         Finally, you should also call your new skill function with proper setting. 
    #         Please start with a reasoning step and then write your code in triplets.

    #         Available skills:
    #         - navigate(object_name): navigate to an 'object_name'
    #         - pick_up(object_name): pick up an 'object_name'
    #         - place_to(object_name): place to 'object_name'

    #         Task: {task}
    #     """
        
        
    #     response = None
        
    #     while response is None:
    #         try:
    #             response = model.generate_content([user_prompt], generation_config=generation_config, safety_settings=safety_settings).text
    #         except Exception as e:
    #             time.sleep(20)
    #     memory.chat_memory.add_ai_message(response)
    #     print("Assistant: " + response)
    #     pattern = r'```.*?\n(.*?)```'
    #     matches = re.findall(pattern, response, re.DOTALL)
            
    #     if matches:
    #         code = '\n'.join(matches).strip()
    #         try:
    #             exec(code, self.safe_globals)
    #         except Exception as e:
    #             memory.chat_memory.add_user_message(f"Task {task} failed with error: {str(e)}")
    #             self.handle_failure_recovery(task)
    #     else:
    #         print("No valid code found.")

    def handle_failure_recovery(self, task):
        recovery_prompt = f"The task '{task}' failed in the previous attempt. Please suggest an alternative approach or parameters to complete the task."
        recovery_response = None
        while recovery_response is None:
            try:
                recovery_response = llm_chain.run(task=sys_prompt)
            except Exception as e:
                time.sleep(20)
        recovery_response = llm_chain.run(task=recovery_prompt)
        print("Recovery Suggestion: " + recovery_response)
        pattern = r'```.*?\n(.*?)```'
        matches = re.findall(pattern, recovery_response, re.DOTALL)
        
        if matches:
            code = '\n'.join(matches).strip()
            try:
                exec(code, self.safe_globals)
            except Exception as e:
                memory.chat_memory.add_user_message(f"Task {task} failed with error: {str(e)}")
                self.handle_failure_recovery(task)
        else:
            print("No valid code found.")
    
    def sub_task_complete(self, task: str) -> None:  
        index = 0
        while True:
            imgs = self.observe(task)      
            genai.configure(api_key=google_api_key_2)
            model = genai.GenerativeModel(model_name=f"models/gemini-1.5-flash-exp-0827")

            user_prompt = f"""
                You are a robot in a dynamically changing environment with wheels and a mechanical arm that can move up and down, stretch out and draw back. 
                You need to complete this task: {task}

                You have these functions available:
                - navigate(object_name): navigate to an object
                - pick_up(object_name): pick up an object, always after navigate(object_name)
                - place_to(object_name): place to a location, always after navigate(object_name)

                Write ONLY the necessary function calls to complete the task.
                DO NOT create new functions or classes.
                Write each action on a new line.
                Pleasse be careful about each object name, it need to be detail but also concise. You could add adjectives before the noun to make it unique as there might be many similar objects in the scene.
                Especially for navigate object, it is better to directly navigate to the object that you want to manipulate. 
                Be careful about object name with human, it's better to depict with adjectives such as their clothes or other traits rather than just say human.
                You should always propose navigate before any pick_up or place_to action

                For example, if the task is "give me a cup", write:
                navigate("blue cup on the table")
                pick_up("blue cup")
                navigate("human in blue coat")
                place_to("human in blue coat")

                Here is your original decomposed tasks and the history for your reference: 
                {self.history}

                Here is actions that you have finished, you should propose new action sequences that will be executed after them:
                {self.task_finished}

                Here is your original plan for next actions:
                {self.task_planned}

                You are holding {self.object_on_hand} on your hand that block you from picking up other objects.

                If you think the task has already been finished, please output nothing

                Your code:
            """
            
            response = None
            while response is None:
                try:
                    response = model.generate_content([user_prompt, *imgs], 
                                                generation_config=generation_config, 
                                                safety_settings=safety_settings).text
                except Exception as e:
                    time.sleep(20)
                    continue

            print("Assistant: " + response)

            code_lines = []
            if '```' in response:
                matches = re.findall(r'```.*?\n(.*?)```', response, re.DOTALL)
                if matches:
                    code = matches[0]
                    code_lines = [line.strip() for line in code.split('\n') 
                                if line.strip() and 
                                not line.strip().startswith('def ') and 
                                not line.strip().startswith('#') and 
                                not line.strip().startswith('class ')]
            elif len(response) < 5:
                return 
            else:
                code_lines = [line.strip() for line in response.split('\n') 
                            if line.strip() and 
                            '(' in line and 
                            ')' in line and
                            not line.strip().startswith('def ') and
                            not line.strip().startswith('#')]
            self.task_planned = code_lines[1:]
            line = code_lines[0]
            try:
                print(f"\nExecuting action: {line}")
                exec(line, self.safe_globals)
                self.task_finished.append(line)
            except Exception as e:
                print(f"Action failed: {line}")
                print(f"Error: {str(e)}")
                self.handle_failure_recovery(task)
                return
            index += 1

    def handle_failure_recovery(self, task):
        recovery_prompt = f"""
        The task '{task}' failed. Write ONLY direct function calls to complete the task.
        Available functions:
        - navigate(object_name)
        - pick_up(object_name)
        - place_to(object_name)
        
        Write each action on a new line. DO NOT create new functions.
        """
        
        recovery_response = None
        while recovery_response is None:
            try:
                recovery_response = llm_chain.run(task=recovery_prompt)
            except Exception as e:
                time.sleep(20)
                continue

        print("Recovery Suggestion: " + recovery_response)
        
        # Extract and execute recovery actions
        code_lines = [line.strip() for line in recovery_response.split('\n') 
                     if line.strip() and 
                     '(' in line and 
                     ')' in line and
                     not line.strip().startswith('def ')]
        
        for line in code_lines:
            try:
                print(f"\nExecuting recovery action: {line}")
                exec(line, self.safe_globals)
            except Exception as e:
                print(f"Recovery action failed: {line}")
                print(f"Error: {str(e)}")


    def simp_task_decompose(self, task): 
        genai.configure(api_key=google_api_key_1)
        user_prompt = f"""You are a robot with wheels and a mechanical arm with a gripper on its end. 
        Your task is to {task}. I will show you the environment around you and you need to split the task into reasonable subtask that could be decomposed into navigation to an object, grasping an object or placing to an object.
        Please be concise, do not propose any unnecessary or repeating subtask. And possibly summarize multiple substasks into on single subtask. 
        
        Note: 
        1. You can utilize any objects you need to reach your goal.
        2. Please propose with the most efficient action sequence. 
        3. You need to give a series of actions, each action or action groups should have the following format in triplets: 
        '''
        Task: describe your subgoal and what actions you need to execute.
        Reasoning: justify why the next action is important to solve the task. 
        '''

        Following is images of the environment around you: 
        """
        model = genai.GenerativeModel(model_name=f"models/gemini-1.5-flash-exp-0827")
        image_paths = glob.glob(os.path.join(self.demo.image_processor.log, "*.jpg"))
        images = []
        for img_path in image_paths:
            img = Image.open(img_path)
            images.append(img)
    
        response = None
        
        while response is None:
            try:
                response = model.generate_content([user_prompt, *images], generation_config=generation_config, safety_settings=safety_settings).text
            except Exception as e:
                time.sleep(20)
        memory.chat_memory.add_ai_message(response)
        print("Assistant: " + response)
        return response

    def observe(self, task):
        genai.configure(api_key=google_api_key_2)
        user_prompt = f"""You are a robot with wheels and a mechanical arm with a gripper on its end.
        Your task is to {task}. You have following available action functions, please answer if you need to observe and examine any objects before action planning.

        Available functions:
        - navigate(object_name)
        - pick_up(object_name)
        - place_to(object_name)
        
        Please directly answer in a list of object name that you want to observe. e.g. [cup, table, chair] 

        Here is the history of your plan and action for your reference: 
        {self.history}
        """
        model = genai.GenerativeModel(model_name=f"models/gemini-1.5-flash-exp-0827")
        response = None
        while response is None:
            try:
                response = model.generate_content([user_prompt], generation_config=generation_config, safety_settings=safety_settings).text
            except Exception as e:
                time.sleep(20)
        memory.chat_memory.add_ai_message(response)
        print("Assistant: " + response)
        object_lst = re.findall(r'\[(.*?)\]', response)[0].split(', ')

        imgs = []
        obs_id_set = set()
        for object_name in object_lst:
            obs_id = self.demo.image_processor.voxel_map_localizer.find_top_obs_id_for_A(object_name)
            for id in obs_id.flatten().tolist():
                obs_id_set.add(str(id))
        for obs_id in obs_id_set:
            imgs.append(Image.open(self.demo.image_processor.log + "/rgb" + obs_id + ".jpg"))
        print('Image Viewed:', obs_id_set)
        return imgs

    def task_complete(self, task):
        tasks = re.findall(r"Task:\s*(.*?)\.", self.simp_task_decompose(task), re.DOTALL)
        self.task_decomposed += f'Your are given the task {task}\nIt has been decomposed into following subtasks: \n'
        self.task_decomposed += '\n'.join(tasks)
        self.task_decomposed += 'Now you are going to propose specific command to complete these tasks'

        self.history = self.task_decomposed

        # for sub_task in tasks:
        #     print(f"\nExecuting subtask: {sub_task}")
        #     self.sub_task_complete(sub_task)
        self.sub_task_complete(task)