#### PromptTemplate
```
from langchain import PromptTemplate  
  
# An example prompt with no input variables  
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")  
no_input_prompt.format()  
# -> "Tell me a joke."  
 
# An example prompt with multiple input variables  
multiple_input_prompt = PromptTemplate(  
input_variables=["adjective", "content"],  
template="Tell me a {adjective} joke about {content}."  
)  
multiple_input_prompt.format(adjective="funny", content="chickens")  
# -> "Tell me a funny joke about chickens."

# 上面可以使用PromptTemplate.from_template使PromptTemplate自动推断input_variables的值

template = "Tell me a {adjective} joke about {content}."

prompt_template = PromptTemplate.from_template(template)
prompt_template.input_variables
# -> ['adjective', 'content']
prompt_template.format(adjective="funny", content="chickens")
# -> Tell me a funny joke about chickens.

# 聊天提示模板
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages()

# 通过pipelinePromptTemplate进行promptTemplate组合
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """You are impersonating {person}."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

start_template = """Now, do this for real!

Q: {input}
A:"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt)
]
pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)

print(pipeline_prompt.format(
    person="Elon Musk",
    example_q="What's your favorite car?",
    example_a="Telsa",
    input="What's your favorite social media site?"
))
```

[[llm_models#聊天模型]]
