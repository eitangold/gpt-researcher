from langchain.prompts import (PromptTemplate,
                               ChatPromptTemplate,
                            )
from langchain_openai import ChatOpenAI
from deepeval.metrics import (AnswerRelevancyMetric,
                              BiasMetric,
                            )   
from deepeval.test_case import LLMTestCase
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai.types.safety_types import (HarmBlockThreshold,
                                                    HarmCategory
                                                    )
from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio
class OpenAiDeepEval(DeepEvalBaseLLM):
    def __init__(self,model):
        self.model = model
        # self.model = self.model
    def get_model_name(self):
        return self.model.model_name
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        chat_model = self.model
        return chat_model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.model
        res = await chat_model.ainvoke(prompt)
        return res.content



class GoogleGemenai(DeepEvalBaseLLM):
    """Class to implement Vertex AI for DeepEval"""

    @staticmethod
    def make_model():
        return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                },
        )
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "gemenai"
    
class LLMJudge:
    def __init__(self):
        self.default_model = ChatOpenAI(model=os.environ.get('OPENAI_LLM_JUDGE_MODEL','gpt-4o-mini'),
                            temperature=0,
                            api_key=os.environ.get('OPENAI_API_KEY'))
        self.llm = OpenAiDeepEval(self.default_model)  # Default LLM, can be changed later
        self.prompts_path = {}

    def set_llm(self, new_llm):
        self.llm = new_llm

    async def answer_relevancy(self, input:str, llm_response:str):
        """
        This fucntion will take input and llm_response and return the relevancy score \n
        calculated by the LIBRARY Deepeval
        #TODO need to check the async_mode to true and get response async
        """
        metric =  AnswerRelevancyMetric(threshold=0.7,
                                            model=self.llm,
                                            include_reason=True,
                                            async_mode=True,
                                        )
        

        test_case = LLMTestCase(input=input, actual_output=llm_response)
        metric.measure(test_case)

        return {'type':'relevancy',
                "reason":metric.reason,
                "numeric":metric.score}
    
    async def answer_bias(self, input:str, llm_response:str):
        """
        This fucntion will take input and llm_response and return the bias score \n
        calculated by the LIBRARY Deepeval
        #TODO need to check the async_mode to true and get response async
        """
        metric =  BiasMetric(threshold=0.7,
                                            model=self.llm,
                                            include_reason=True,
                                            async_mode=True,
                                        )
        

        test_case = LLMTestCase(input=input, actual_output=llm_response)
        metric.measure(test_case)
        return {'type':'bias',
                "reason":metric.reason,
                "numeric":metric.score}
    
# async def call_metrics():
#     j = LLMJudge()
#     question = "hey what is my name"
#     response = 'my name'
#     async with asyncio.TaskGroup() as tg:
#         results = [tg.create_task(j.answer_relevancy(question,response)),
#                    tg.create_task(j.answer_bias(question,response))]
    
#     # results = await asyncio.gather(*a)
#     return results
    
# async def main():
#     results = await call_metrics()
#     print(results)

# asyncio.run(main())
    