> 🌟 该文档基于文档创建时最新稳定版 0.3.0: https://github.com/Significant-Gravitas/Auto-GPT/tree/v0.3.0

## AutoGPT 是什么

AutoGPT 是 GitHub 上的一个开源项目，它致力于使 GPT4 完全自主。用户使用 AutoGPT，只需要告诉 AutoGPT 一个目标，AutoGPT 会自主生成执行计划，自主和 GPT4 或者 GPT3.5 交互，并一步一步完成计划，最后输出用户想要的结果，整个过程完全不需要用户参与。另外，AutoGPT 实现了很多工具，可以进行网络搜索、文件操作、代码执行等操作，和现实世界打通，极大扩展了 ChatGPT 的能力

  

由于不需要用户引导 AI，项目一经推出就爆火，截至目前，项目上线五十多天，拥有 13.1w star，2.67w fork，是 GitHub 历史上增长速度最快的项目之一。前特斯拉总监、刚刚回归 OpenAI 的 Andrej Karpathy 也为其大力宣传，称「AutoGPT 是 prompt 工程的下一个前沿。」

![](https://bytedance.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWM2MjViN2Y1MGRhNDJiN2RkZjk1Nzk2MTZmM2ZhMjBfNGhkcERBbVdpN0VTVkVCNGpqWk1qRDJkSWZidGJ2eFZfVG9rZW46SFVCRWJZdzByb3VKUW94VHJseGM0OXVJbmNIXzE2ODQ3MzIzMTg6MTY4NDczNTkxOF9WNA)

  

和openai官方的ChatGPT/GPT4相比，AutoGPT主要省去了一步步和ChatGPT/GPT4交互的过程，隐藏了prompt等细节；同时AutoGPT实现了多个工具，并通过prompt告诉大模型工具的能力，使大模型可以使用工具，扩展大模型的能力边界

  

和 langchain 相比，langchain 是一个库，需要编程后才能和大模型交互，AutoGPT 是一个应用，可以直接和大模型交互；从概念上来讲，AutoGPT 相当于 langchain 中的 Agent，AutoGPT 中的工具相当于 langchain 中的 tool，AutoGPT 中的 memory 和 langchain 的 memory 相当

  

AutoGPT 的官方站点：

-   GitHub：https://github.com/Significant-Gravitas/Auto-GPT
    
-   网站：https://agpt.co
    
-   文档
    
    -   https://docs.agpt.co
        
    -   https://github.com/Significant-Gravitas/Auto-GPT/tree/v0.3.0/docs
        

## AutoGPT 使用

文档：https://docs.agpt.co/setup

### 配置

在使用之前，需要先完成配置，AutoGPT 支持 openai 和 azure 的 GPT3.5/GPT4 模型

-   openai
    
    -   .env 中配置 OPENAI_API_KEY
    
-   azure
    
    -   开启 Azure：.env 中设置 USE_AZURE=True
        
    -   配置 key：.env 中配置 OPENAI_API_KEY，该 key 是 azure 的 key，和 openai 无关
        
    -   配置 azure 模型信息：azure.yaml 中配置
        
        -   azure_api_base：azure 的 api base
            
        -   azure_api_version：azure 的 api 版本
            
        -   fast_llm_model_deployment_id：azure 中部署的大模型，如果需要 AutoGPT 使用 GPT3.5，写入 GPT3.5 模型 id；如果需要 AutoGPT 用 GPT4，写入 GPT4 的模型 id
            
        -   smart_llm_model_deployment_id：azure 中部署的大模型，id 选择同上，和以上保持一致即可
            
        -   embedding_model_deployment_id：azure 中部署的 embeding 模型，是必选项，一般是 text-embedding-ada-002 模型
            

  

以上是大模型相关配置，也是运行 AutoGPT 必须的配置。除此之外，AutoGPT 还支持工具、memory 相关配置，比如可以配置使用 Stable Diffusion WebUI 生成图片，详细可以在文档中查看

### 运行

AutoGPT 可以在 docker 中和 docker 外运行

-   docker 内

AutoGPT 已经构建好 docker 镜像，可以直接拉下来使用

官方文档中提供了 docker-compose 配置文件，其中做好了文件挂载等工作，直接运行即可

-   docker 外

需要提前安装 python3 依赖，依赖位于 requirements.txt

运行 AutoGPT 项目根目录的 run.sh

  

AutoGPT 提供了多个命令行选项，控制运行行为，主要选项如下：

| 选项             | 作用                                                         |
| ---------------- | ------------------------------------------------------------ |
| --help           | 列出并解释所有可用选项                                       |
| --continuous/ -c | 开启连续模式，开启之后，会直接执行大模型选择的命令，不需要用户确认 |
| --gpt3only       | 只使用 GPT3.5，0.3.0 版本，对 openai 有效，对 azure，使用的模型是 azure 配置中的模型，不受影响 |
| --gpt4only       | 只使用 GPT4，0.3.0 版本，对 openai 有效，对 azure，使用的模型是 azure 配置中的模型，不受影响 |
| --debug          | 开启 debug 模式，输出 debug 日志                                 |

项目运行完成，结果在项目根目录的`autogpt/auto_gpt_workspace`中

```Plain
🎆 AutoGPT Web版已经在路上，目前可以在官网提供邮件加入waitlist
```



## 工作流程

启动 AutoGPT 之后，会提示用户输入需要 GPT 做的事，输入任务后，AutoGPT 首先会询问大模型，把任务分解为几个目标，之后会开启一个循环，一步步自主完成用户的指定的任务

AutoGPT 执行主要流程如下，其中省略了部分分支

![](assents/Pasted%20image%2020230522132003.png)
## Prompt

Prompt 是和大模型交互的唯一方式，Prompt 的质量直接影响大模型的输出，可以说是使用大模型时最重要的部分。本节从多个场景分析 AutoGPT 如何构造 Prompt

### 分解任务

用户输入任务后，AutoGPT 会和大模型交互，请求一个 AI 名、一个 AI 角色描述和至多 5 个目标

相关 Prompt 如下，其中 {user_prompt} 为用户输入任务

```json
{
    "role": "system",
    "content": """
Your task is to devise up to 5 highly effective goals and an appropriate role-based name (_GPT) for an autonomous agent, ensuring that the goals are optimally aligned with the successful completion of its assigned task.

The user will provide the task, you will provide only the output in the exact format specified below with no explanation or conversation.

Example input:
Help me with marketing my business

Example output:
Name: CMOGPT
Description: a professional digital marketer AI that assists Solopreneurs in growing their businesses by providing world-class expertise in solving marketing problems for SaaS, content products, agencies, and more.
Goals:
- Engage in effective problem-solving, prioritization, planning, and supporting execution to address your marketing needs as your virtual Chief Marketing Officer.

- Provide specific, actionable, and concise advice to help you make informed decisions without the use of platitudes or overly wordy explanations.

- Identify and prioritize quick wins and cost-effective campaigns that maximize results with minimal time and budget investment.

- Proactively take the lead in guiding you and offering suggestions when faced with unclear information or uncertainty to ensure your marketing strategy remains on track.
"""
},
{
    "role": "user",
    "content": f"Task: '{user_prompt}'\nRespond only with the output in the exact format specified in the system prompt, with no explanation or conversation.\n",
},
```

### 历史对话摘要

由于历史对话长度很长，超过大模型允许的 token 数量，AutoGPT 把历史对话进行摘要后发送

相关 Prompt 如下，其中 {current_memory} 是之前对话摘要，{new_events} 是本轮新增对话内容

```json
{
    "role": "user",
    "content": f'''Your task is to create a concise running summary of actions and information results in the provided text, focusing on key and potentially important information to remember.

You will receive the current summary and the your latest actions. Combine them, adding relevant key information from the latest development in 1st person past tense and keeping the summary concise.

Summary So Far:
"""
{current_memory}
"""

Latest Development:
"""
{new_events}
"""
'''
}
```

### 询问下一步动作

AutoGPT 运行采用 step-by-step 的形式，每次都会询问大模型下一步动作，得到动作后本地执行，在把执行结果反馈给大模型，继续询问下一步动作。询问下一步动作的 Prompt 比较复杂，可以分为多个部分，分别如下

#### system prompt

system prompt 示例如下，也可以分成几部分：

-   分解任务之后获得的 AI 名、AI 角色和任务目标
    
-   限制
    
-   支持的命令
    
-   可以提供的资源或者能力
    
-   性能评估
    
-   输出描述
    

```Python
"system": """You are WeatherGPT_CN, 一个智能气象助手，为您提供准确、实时的天气信息，帮助您更好地规划活动和出行。
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. 以中文为主要语言，为您提供清晰、简洁的天气预报。
2. 获取并分析来自权威数据源的实时天气信息，确保准确性和可靠性。
3. 针对您的具体需求，提供详细的天气信息，包括温度、湿度、风向、风速等。
4. 根据您的地理位置，为您提供最相关的天气预报。
5. 及时更新天气信息，确保您随时了解最新的天气状况。


Constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"

Commands:
1. analyze_code: Analyze Code, args: "code": "<full_code_string>"
2. execute_python_file: Execute Python File, args: "filename": "<filename>"
3. append_to_file: Append to file, args: "filename": "<filename>", "text": "<text>"
4. delete_file: Delete file, args: "filename": "<filename>"
5. list_files: List Files in Directory, args: "directory": "<directory>"
6. read_file: Read file, args: "filename": "<filename>"
7. write_to_file: Write to file, args: "filename": "<filename>", "text": "<text>"
8. google: Google Search, args: "query": "<query>"
9. improve_code: Get Improved Code, args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
10. send_tweet: Send Tweet, args: "tweet_text": "<tweet_text>"
11. browse_website: Browse Website, args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
12. write_tests: Write Tests, args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
13. delete_agent: Delete GPT Agent, args: "key": "<key>"
14. get_hyperlinks: Get text summary, args: "url": "<url>"
15. get_text_summary: Get text summary, args: "url": "<url>", "question": "<question>"
16. list_agents: List GPT Agents, args: () -> str
17. message_agent: Message GPT Agent, args: "key": "<key>", "message": "<message>"
18. start_agent: Start GPT Agent, args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
19. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"

Resources:
1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

Performance Evaluation:
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
5. Write all code to a file.

You should only respond in JSON format as described below 
Response Format: 
{
    "thoughts": {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\n- list that conveys\n- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args": {
            "arg name": "value"
        }
    }
} 
Ensure the response can be parsed by Python json.loads
"""
```

#### context prompt

context 部分是对话需要的上下文，示例如下。也可以分为几部分

-   当前时间
    
-   历史所有对话和命令执行结果摘要
    
-   最近的对话和命令执行结果
    

```Python
{
    "system": "The current time and date is Tue May 16 11:39:01 2023",
    "system": '''This reminds you of these events from your past: 
I was created and googled the real-time weather in Shanghai. I found several websites with weather information, including the Shanghai Meteorological Bureau and The Weather Channel.''',
    "user": "Determine which next command to use, and respond using the format specified above:",
    "assistant": '''{
    "thoughts": {
        "text": "我需要查找上海的实时天气信息。",
        "reasoning": "用户需要了解上海的实时天气状况。",
        "plan": "- 使用browse_website命令\n- 访问weather.com.cn网站\n- 查询上海的实时天气",
        "criticism": "无",
        "speak": "我将查找上海的实时天气信息。"
    },
    "command": {
        "name": "browse_website",
        "args": {
            "url": "http://www.weather.com.cn/weather/101020100.shtml",
            "question": "上海现在的天气情况是怎样的？"
        }
    }
}''',
    "system": "Command browse_website returned: ('Error: timeout: Timed out receiving message from renderer: 299.610', None)",
}
```

#### trigger prompt

trigger prompt 用来引导大模型进行输出，示例如下

```Python
    "user": "Determine which next command to use, and respond using the format specified above:"
```

## Command

AutoGPT 的 command 概念可以类比 langchain 的 tool，可以执行特定的任务。AutoGPT 内置了很多 command，汇总如下

| 分类                | command 名              | 作用                                                         | 工作原理                                            |
| ------------------- | ---------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| 代码                | analyze_code           | 分析 python 代码，提出提升建议                                 | 询问大模型                                          |
| execute_python_file | 执行 python 代码         | 拉取 docker 镜像，在容器内执行                                 |                                                     |
| improve_code        | 根据建议，改写代码     | 询问大模型                                                   |                                                     |
| write_tests         | 为 python 代码写测试     | 询问大模型                                                   |                                                     |
| 文件                | append_to_file         | 追加内容到文件                                               | -                                                   |
| delete_file         | 删除文件               | -                                                            |                                                     |
| list_files          | 列出目录下文件         | -                                                            |                                                     |
| read_file           | 读文件                 | -                                                            |                                                     |
| write_to_file       | 写文件                 | -                                                            |                                                     |
| 网络                | google                 | 使用搜索引擎，搜索内容                                       | 支持两种搜索：google api 搜索，需要 google keyddg 搜索 |
| browse_website      | 请求网页，获取网页内容 | 使用 selenium 请求网页，使用 BeautifulSoup 解析网页内容，解析内容后询问大模型，从网页内容中获取结果 |                                                     |
| get_hyperlinks      | 从 url 中获取超链接      | 通过 requests 请求网页，使用 BeautifulSoup 解析网页内容          |                                                     |
| gpt agent           | start_agent            | 创建 agent                                                    | agent 可以和大模型交互，有独立的上下文               |
| message_agent       | 使用 agent 和大模型交互  | -                                                            |                                                     |
| list_agents         | 列出所有 agent          | -                                                            |                                                     |
| delete_agent        | 删除 agent              | -                                                            |                                                     |
| 其他                | get_text_summary       | 获取文本摘要                                                 | 询问大模型                                          |
| send_tweet          | 发送 tweeter            | 使用 tweeter api                                              |                                                     |
| task_complete       | 结束任务               | -                                                            |                                                     |



## Memory

由于和大模型交互时，多次请求之间没有关联，需要额外的组件管理一次会话的多轮对话信息。此外，一次请求的 token 数量有上限，目前 GPT3.5 允许最多 4K，GPT4 最多允许 8K 或者 32K，当上下文过长时，需要额外处理。AutoGPT**通过 memory 处理以上问题**

### Memory 类型

AutoGPT 支持多种 Memory 类型，具体如下

| 类型     | 描述               | 备注                                                         |
| -------- | ------------------ | ------------------------------------------------------------ |
| local    | 本地 json 文件       | docker 外运行时默认                                           |
| redis    | redis 服务器        | 使用 docker 运行时默认；可以跨多次启动，其他 memory 在启动时会清空 |
| pinecone | 一种向量数据库     | -                                                            |
| milvus   | 一种开源向量数据库 | -                                                            |
| weaviate | 一种开源向量数据库 | -                                                            |

### 上下文处理

0.3.0 版本，会把所有历史对话和命令执行结果做摘要，并把摘要加入 Prompt

  

0.2.2 版本，会对每一轮对话做 embeding，并把对话和 embeding 存入 memory。询问大模型下一步动作时，会把最近几轮对话最相关的内容从 memory 中召回，在 token 足够时，尽可能多的加入召回的内容

### 文件 embeding

AutoGPT 支持把多个文件做 embeding 后加入 memory，具体 embeding 的方式是固定大小做 chunk，并保留一定 overlap

  

在项目根目录，直接执行 `python data_ingestion.py -h` 可以获取更多信息

  

## plugin

AutoGPT 通过插件扩展能力。在 AutoGPT 执行流程中，多处都增加了钩子，插件可以在任意钩子处执行。通过插件，用户可以增加自定义命令、修改 prompt、处理大模型输出或执行其他自定义逻辑

  

用户可以通过继承 [AutoGPTPluginTemplate](https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template/blob/master/src/auto_gpt_plugin_template/__init__.py#L15)实现插件，只需要实现对应的函数即可。但当前的插件协议并不固定，需要注意 AutoGPT 官方插件维护在仓库 https://github.com/Significant-Gravitas/Auto-GPT-Plugins 中，其中包含一个百度搜索插件，实现位于 [baidu_search](https://github.com/Significant-Gravitas/Auto-GPT-Plugins/tree/master/src/autogpt_plugins/baidu_search)。

  

插件的使用方式可以参考上述官方仓库 README 文件

## 效果和展望

### Benchmark

AutoGPT 官方关注到 benchmark 问题，开启了新项目[Auto-GPT-Benchmarks](https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks)。该项目使用 OpenAI 提供的 Evals 框架，使用 OpenAI benchmark 数据

  

该项目最近才开始，目前还没有看到相关数据

### 实际使用

综合实际使用，以及网络和社交平台的信息，AutoGPT 的效果并不令人十分满意，多数结果不及预期，还有比较大的提升空间

### 展望

虽然目前 AutoGPT 效果不尽如人意，但它第一次把想象中完全自动化的 AI 助手带入现实。目前项目发展迅速，随着大模型的能力不断增强，以及 command 能力的不断细化，AutoGPT 的未来很值得期待

## Reference

https://github.com/Significant-Gravitas/Auto-GPT

https://github.com/Significant-Gravitas/Auto-GPT/tree/master/docs

https://docs.agpt.co

https://news.agpt.co

https://github.com/Significant-Gravitas/Auto-GPT-Benchmarks

https://github.com/openai/evals/tree/main

https://github.com/Significant-Gravitas/Auto-GPT-Plugins

https://github.com/Significant-Gravitas/Auto-GPT-Plugin-Template

https://www.51cto.com/article/751884.html

https://juejin.cn/post/7222845337492799548