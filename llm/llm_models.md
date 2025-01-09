### 模型
#### llms
将文本字符串作为输入，并返回文本字符串作为输出。
##### 基础调用
```
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-ada-001", n=2, best_of=2)
llm("Tell me a joke")
#列表传输
llm_result = llm.generate(["Tell me a joke", "Tell me a poem"]*15)
len(llm_result.generations)
llm_result.generations[0]

#token使用数量
print(llm_result.llm_output)
llm.get_num_tokens("what a joke")
 
```

##### 异步llms
```
import time
import asyncio
 
from langchain.llms import OpenAI
 
def generate_serially():
    llm = OpenAI(temperature=0.9)
    for _ in range(10):
        resp = llm.generate(["Hello, how are you?"])
        print(resp.generations[0][0].text)
 
async def async_generate(llm):
    resp = await llm.agenerate(["Hello, how are you?"])
    print(resp.generations[0][0].text)
 
async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    tasks = [async_generate(llm) for _ in range(10)]
    await asyncio.gather(*tasks)
 
s = time.perf_counter()
# If running this outside of Jupyter, use asyncio.run(generate_concurrently())
await generate_concurrently() 
elapsed = time.perf_counter() - s
print('\033[1m' + f"Concurrent executed in {elapsed:0.2f} seconds." + '\033[0m')
 
s = time.perf_counter()
generate_serially()
elapsed = time.perf_counter() - s
print('\033[1m' + f"Serial executed in {elapsed:0.2f} seconds." + '\033[0m')

```

##### 创建自定义的llm
自定义llm需要提供一个_call函数:接收字符串,一些可选的停用词,并返回一个字符串
可以提供一个可选的_identifying_params属性，用于输出类时的显示，应该是返回一个字典
```
from typing import Any, List, Mapping, Optional 
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class CustomLLM(LLM):
 
    n: int
 
    @property
    def _llm_type(self) -> str:
        return "custom"
 
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return prompt[:self.n]
 
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
 """Get the identifying parameters."""
        return {"n": self.n}


llm = CustomLLM(n=10)
# 执行llm,返回前10个字符串
llm("This is a foobar thing")
#返回_identifying_params函数的结果
print(llm)
```


##### 创建虚假的llm
模拟LLM以特定方式响应时会发生什么,直接设置返回结果
```
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

tools = load_tools(["python_repl"])
responses=[
    "Action: Python REPL\nAction Input: print(2 + 2)",
    "Final Answer: 4"
]
llm = FakeListLLM(responses=responses)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.run("whats 2 + 9")
```

##### 缓存llm调用
把llm的结果进行缓存，有多种方式
```
# 内存缓存
import langchain
from langchain.cache import InMemoryCache
langchain.llm_cache = InMemoryCache()
# To make the caching really obvious, lets use a slower model.
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)

# sqlite缓存
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# redis缓存
# We can do the same thing with a Redis cache
# (make sure your local Redis instance is running first before running this example)
from redis import Redis
from langchain.cache import RedisCache
langchain.llm_cache = RedisCache(redis_=Redis())

# 语义缓存 使用redis进行向量存储缓存，并根据语义相似性评估命中率。
from langchain.embeddings import OpenAIEmbeddings
from langchain.cache import RedisSemanticCache
 
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings()
)

# 使用gptcache进行缓存，用于存储和检索 LLM 的响应结果，从而减少在处理相同或相似输入时重复调用 LLM 所带来的延迟和计算资源消耗
import gptcache
from gptcache.processor.pre import get_prompt
from gptcache.manager.factory import get_data_manager
from langchain.cache import GPTCache
 
# Avoid multiple caches using the same file, causing different llm model caches to affect each other
i = 0
file_prefix = "data_map"
 
def init_gptcache_map(cache_obj: gptcache.Cache):
    global i
    cache_path = f'{file_prefix}_{i}.txt'
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=get_data_manager(data_path=cache_path),
    )
    i += 1
 
langchain.llm_cache = GPTCache(init_gptcache_map)

# sqlalchemy缓存
from langchain.cache import SQLAlchemyCache
from sqlalchemy import create_engine
engine =create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine)

# 自定义sqlalchemy模式
from sqlalchemy import Column, Integer, String, Computed, Index, Sequence
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import TSVectorType
from langchain.cache import SQLAlchemyCache
 
Base = declarative_base()
 
class FulltextLLMCache(Base):  # type: ignore
 """Postgres table for fulltext-indexed LLM Cache"""
 
    __tablename__ = "llm_cache_fulltext"
    id = Column(Integer, Sequence('cache_id'), primary_key=True)
    prompt = Column(String, nullable=False)
    llm = Column(String, nullable=False)
    idx = Column(Integer)
    response = Column(String)
    prompt_tsv = Column(TSVectorType(), Computed("to_tsvector('english', llm || ' ' || prompt)", persisted=True))
    __table_args__ = (
        Index("idx_fulltext_prompt_tsv", prompt_tsv, postgresql_using="gin"),
    )
 
engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")
langchain.llm_cache = SQLAlchemyCache(engine, FulltextLLMCache)

# 关闭缓存
llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2, cache=False)

# 链式缓存,关闭链中特定节点的缓存
llm = OpenAI(model_name="text-davinci-002")
no_cache_llm = OpenAI(model_name="text-davinci-002", cache=False)

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
 
text_splitter = CharacterTextSplitter()
with open('../../../state_of_the_union.txt') as f:
    state_of_the_union = f.read()
texts = text_splitter.split_text(state_of_the_union)

from langchain.docstore.document import Document
docs = [Document(page_content=t) for t in texts[:3]]
from langchain.chains.summarize import load_summarize_chain

chain = load_summarize_chain(llm, chain_type="map_reduce", reduce_llm=no_cache_llm)
```

##### 序列化llm
llms可以用json或者yaml格式保存在磁盘上
```
# 读取llm配置文件
llm = load_llm("llm.yaml")

# 保存llm配置
llm.save("llm.yaml")
```

##### stram llm
让结果一部分一部分显示，不用等待，减少延迟感知
```
from langchain.llms import OpenAI, Anthropic
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
 

llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = llm("Write me a song about sparkling water.")
 
```

##### 令牌使用跟踪
```
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-002", n=2, best_of=2)
with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
  
```
#### 聊天模型
将聊天消息列表作为输入，并返回聊天消息
LangChain目前支持的消息类型有`AIMessage`、`HumanMessage`、`SystemMessage`和`ChatMessage` - `ChatMessage`接受任意角色参数。
##### 基础调用
```
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
chat = ChatOpenAI(temperature=0)
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])

messages = [ SystemMessage(content="You are a helpful assistant that translates English to French."), HumanMessage(content="I love programming.")]
chat(messages)

# 使用generate进行多组messages
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result

# prompt_templates
使用MessagePromptTemplate创建模板,再通过一个或多个模板构建ChatPromptTemplate,用`ChatPromptTemplate`的`format_prompt` - 这将返回一个`PromptValue`，您可以将其转换为字符串或消息对象
template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
 
# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())

chain = LLMChain(llm=chat, prompt=chat_prompt)

# 流处理
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
chat = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0)
resp = chat([HumanMessage(content="Write me a song about sparkling water.")])
 

# 少样本实例 通过较少的示例理解并完成任务
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates english to pirate."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("Argh me mateys")
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)
# get a chat completion from the formatted messages
chain.run("I love programming.")
```

#### 文本嵌入模型
文本作为输入，并返回一个浮点数列表
```
# Aleph Alpha嵌入，分为对称嵌入和不对称嵌入
# 不对称嵌入
from langchain.embeddings import AlephAlphaAsymmetricSemanticEmbedding

document = "This is a content of the document"
query = "What is the contnt of the document?"
embeddings = AlephAlphaAsymmetricSemanticEmbedding()
doc_result = embeddings.embed_documents([document])
query_result = embeddings.embed_query(query)

# 对称嵌入
from langchain.embeddings import AlephAlphaSymmetricSemanticEmbedding

text = "This is a test text"
embeddings = AlephAlphaSymmetricSemanticEmbedding()
doc_result = embeddings.embed_documents([text])
query_result = embeddings.embed_query(text)
 
 
```