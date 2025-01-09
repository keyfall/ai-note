### agent

#### demo
```
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
 
 llm = OpenAi(temperature=0)
 tools = load_tools(["serpapi", "llm-math"], llm=llm)
 agent = initialize_agent(tools, llm, agent.AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
 agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
 
```

#### agent_type
zero-shot-react-description  
此代理使用react框架,仅基于工具的描述来确定使用的工具

react-docstore  
使用react与文档存储进行交互  
必须提供两个工具：Search,Lookup

Search工具是搜索文档  
Lookup查找最近找到的文档

self-ask-with-search  
查找问题的事实性答案  
需要使用IntermediateAnswer工具

conversional-react-description  
用于对话环境

#### 自定义代理
代理由两部分组成：工具，代理执行器

```
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain import OpenAI, SerpAPIWrapper
 
 search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
        return_direct=True
    )
]

from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
 
class FakeAgent(BaseSingleActionAgent):
    """Fake Custom Agent."""
    
    @property
    def input_keys(self):
        return ["input"]
    # 返回一个A
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
 
        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.
 
        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")
 
    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.
 
        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.
 
        Returns:
            Action specifying what tool to use.
        """
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")
 
 agent = FakeAgent()
 agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
 
 agent_executor.run("How many people live in canada as of 2023?")
```

#### 带工具检索的自定义代理
```
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
search_tool = Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
def fake_func(inp: str) -> str:
    return "foo"
fake_tools = [
    Tool(
        name=f"foo-{i}", 
        func=fake_func, 
        description=f"a silly function that you can use to get more information about the number {i}"
    ) 
    for i in range(99)
]
ALL_TOOLS = [search_tool] + fake_tools
 
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
 
vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

retriever = vector_store.as_retriever()

def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in docs]

get_tools("whats the weather?")


# Set up the base template
template = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:
 
{tools}
 
Use the following format:
 
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
 
Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s
 
Question: {input}
{agent_scratchpad}"""
 
 from typing import Callable
# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    ############## NEW ######################
    # The list of tools available
    tools_getter: Callable
 
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        ############## NEW ######################
        tools = self.tools_getter(kwargs["input"])
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
        return self.template.format(**kwargs)
 
 prompt = CustomPromptTemplate(
    template=template,
    tools_getter=get_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
 
class CustomOutputParser(AgentOutputParser):
 
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
 
 output_parser = CustomOutputParser()
 
llm = OpenAI(temperature=0)
llm_chain = LLMChain(llm=llm, prompt=prompt)
 
tools = get_tools("whats the weather?")
tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
 
```


