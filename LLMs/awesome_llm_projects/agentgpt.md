# AgentGPT
- [AgentGPT](#agentgpt)
  - [特性](#特性)
  - [基本原理](#基本原理)
    - [行动](#行动)
    - [分析行动结果](#分析行动结果)
    - [执行任务](#执行任务)
    - [整体循环](#整体循环)
    - [memory 的作用](#memory-的作用)

>GPT 自我 Prompt 
>GitHub 地址：https://github.com/reworkd/AgentGPT
>Web 地址：https://agentgpt.reworkd.ai/

## 特性

- 通过向量数据库实现长期记忆 🧠
- 通过 LangChain 实现网页浏览功能 🌐
- 与网站和人交互 👨‍👩‍👦
- 通过文档 API 实现书写功能 📄
- 保存代理运行 💾
- 用户和认证 🔐
- Stripe 集成实现收费下限版本 (所以我们不用再担心基础设施成本) 💵

## 基本原理

![agentgpt 流程图](assents/agentgpt流程图.png)

AgentGPT 不是简单地要求 ChatGPT 创建代码，而是将 OpenAI 的 GPT 模型的多个实例链接在一起，让多个 GPT 代理协同工作，使其能够在没有帮助的情况下完成任务、编写和调试代码以及纠正自己的编写错误等事情。

AgentGPT 实现：行动 -> 分析行动结果 (thinking) -> 执行任务 -> 决定下一步行动的自我循环

### 行动

```yaml
ai_goals:
- 告诉我LangChain如何使用.
- 帮我安装LangChain.
- 告诉我一些可以直接在我电脑上运行LangChain的例子.
ai_name: 我是 GPT
ai_role: LangChain 如何使用
```

AgentGPT prompt 将问题转换为 Prompt 模板，然后加上一些限制条件 CONSTRAINTS，COMMANDS，RESOURCES 和 PERFORMANCE EVALUATION 模板；最后给告诉 ChatGPT 提供一个 JSON 的返回格式，并且需要 python json.loads 可以直接执行的

```json
You are 我是GPT, LangChain如何使用
Your decisions must always be made independently without seeking user assistance. Play to your strengths as an LLM and pursue simple strategies with no legal complications.

GOALS:

1. 告诉我LangChain如何使用.
2. 帮我安装LangChain.
3. 告诉我一些可以直接在我电脑上运行LangChain的例子.
4. 所有的问题都用中文回答.

CONSTRAINTS:

1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. No user assistance
4. Exclusively use the commands listed in double quotes e.g. "command name"

COMMANDS:

1. Google Search: "google", args: "input": "<search>"
5. Browse Website: "browse_website", args: "url": "<url>", "question": "<what_you_want_to_find_on_website>"
6. Start GPT Agent: "start_agent",  args: "name": "<name>", "task": "<short_task_desc>", "prompt": "<prompt>"
7. Message GPT Agent: "message_agent", args: "key": "<key>", "message": "<message>"
8. List GPT Agents: "list_agents", args: ""
9. Delete GPT Agent: "delete_agent", args: "key": "<key>"
10. Write to file: "write_to_file", args: "file": "<file>", "text": "<text>"
11. Read file: "read_file", args: "file": "<file>"
12. Append to file: "append_to_file", args: "file": "<file>", "text": "<text>"
13. Delete file: "delete_file", args: "file": "<file>"
14. Search Files: "search_files", args: "directory": "<directory>"
15. Evaluate Code: "evaluate_code", args: "code": "<full_code_string>"
16. Get Improved Code: "improve_code", args: "suggestions": "<list_of_suggestions>", "code": "<full_code_string>"
17. Write Tests: "write_tests", args: "code": "<full_code_string>", "focus": "<list_of_focus_areas>"
18. Execute Python File: "execute_python_file", args: "file": "<file>"
19. Task Complete (Shutdown): "task_complete", args: "reason": "<reason>"
20. Generate Image: "generate_image", args: "prompt": "<prompt>"
21. Do Nothing: "do_nothing", args: ""

RESOURCES:

1. Internet access for searches and information gathering.
2. Long Term memory management.
3. GPT-3.5 powered Agents for delegation of simple tasks.
4. File output.

PERFORMANCE EVALUATION:

1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.

You should only respond in JSON format as described below

RESPONSE FORMAT:
{
    "thoughts":
    {
        "text": "thought",
        "reasoning": "reasoning",
        "plan": "- short bulleted\
- list that conveys\
- long-term plan",
        "criticism": "constructive self-criticism",
        "speak": "thoughts summary to say to user"
    },
    "command": {
        "name": "command name",
        "args":{
            "arg name": "value"
        }
    }
}

Ensure the response can be parsed by Python json.loads
```

然后使用该内容调用 ChatGPT 借口：

```python
assistant_reply = create_chat_completion(
    model=model,
    messages=current_context,
    max_tokens=tokens_remaining,
)

# Update full message history
full_message_history.append(
    create_chat_message(
        "user", user_input))
full_message_history.append(
    create_chat_message(
        "assistant", assistant_reply))
```

### 分析行动结果

```json
{
    "thoughts":{
        "text": "LangChain is a programming language that can be used for blockchain development. To use it, we need to first download and install it on our computer. After installation, we can run some code examples to test it out. ",
        "reasoning": "I reviewed the purpose and usage of LangChain and suggested a plan for installation and testing",
        "plan": "- Download and install LangChain.\n- Run code examples to test it out.",
        "criticism": "I should have provided more specific detailed steps on how to download and install LangChain",
        "speak": "We need to download and install LangChain before we can use it. Do you have any experience with this?"
    },
    "command":{
        "name": "browse_website",
        "args":{
            "url": "https://langchain.org/docs/getting-started/",
            "question": "How to download and install LangChain?"
        }
    }
}
```

可以看到返回的 plan 拆分为两个步骤：

```text
-  Download and install LangChain.
-  Run code examples to test it out.
```

然后也给出了命令：browse_website 和参数。

### 执行任务

可以从 AgentGPT 的源码中看到，提供了很多执行脚本的方式，比如 google 搜索，执行 execute_shell 脚本等。

```Python
try:
    if command_name == "google":

        # Check if the Google API key is set and use the official search method
        # If the API key is not set or has only whitespaces, use the unofficial search method
        if cfg.google_api_key and (cfg.google_api_key.strip() if cfg.google_api_key else None):
            return google_official_search(arguments["input"])
        else:
            return google_search(arguments["input"])
    elif command_name == "memory_add":
        return memory.add(arguments["string"])
    elif command_name == "start_agent":
        return start_agent(
            arguments["name"],
            arguments["task"],
            arguments["prompt"])
    elif command_name == "message_agent":
        return message_agent(arguments["key"], arguments["message"])
    elif command_name == "list_agents":
        return list_agents()
    elif command_name == "delete_agent":
        return delete_agent(arguments["key"])
    elif command_name == "get_text_summary":
        return get_text_summary(arguments["url"], arguments["question"])
    elif command_name == "get_hyperlinks":
        return get_hyperlinks(arguments["url"])
    elif command_name == "read_file":
        return read_file(arguments["file"])
    elif command_name == "write_to_file":
        return write_to_file(arguments["file"], arguments["text"])
    elif command_name == "append_to_file":
        return append_to_file(arguments["file"], arguments["text"])
    elif command_name == "delete_file":
        return delete_file(arguments["file"])
    elif command_name == "search_files":
        return search_files(arguments["directory"])
    elif command_name == "browse_website":
        return browse_website(arguments["url"], arguments["question"])
    # TODO: Change these to take in a file rather than pasted code, if
    # non-file is given, return instructions "Input should be a python
    # filepath, write your code to file and try again"
    elif command_name == "evaluate_code":
        return ai.evaluate_code(arguments["code"])
    elif command_name == "improve_code":
        return ai.improve_code(arguments["suggestions"], arguments["code"])
    elif command_name == "write_tests":
        return ai.write_tests(arguments["code"], arguments.get("focus"))
    elif command_name == "execute_python_file":  # Add this command
        return execute_python_file(arguments["file"])
    elif command_name == "execute_shell":
        if cfg.execute_local_commands:
            return execute_shell(arguments["command_line"])
        else:
            return "You are not allowed to run local shell commands. To execute shell commands, EXECUTE_LOCAL_COMMANDS must be set to 'True' in your config. Do not attempt to bypass the restriction."
    elif command_name == "generate_image":
        return generate_image(arguments["prompt"])
    elif command_name == "do_nothing":
        return "No action performed."
    elif command_name == "task_complete":
        shutdown()
    else:
        return f"Unknown command '{command_name}'. Please refer to the 'COMMANDS' list for available commands and only respond in the specified JSON format."
# All errors, return "Error: + error message"
except Exception as e:
    return "Error: " + str(e)
```

根据以上给出的 browse_website，参数是`https://langchain.org/docs/getting-started/`和`How to download and install LangChain?`
根据以上的代码就会执行：

```text
browse_website(arguments["url"], arguments["question"])
```

由于`https://langchain.org/docs/getting-started/`这个地址就是错误的，所以执行报错：

```json
User: Determine which next command to use, and respond using the format specified above:

Assistant: {
    "thoughts": {
        "text": "LangChain is a programming language that can be used for blockchain development. To use it, we need to first download and install it on ou r computer. After installation, we can run some code examples to test it out.",
		"reasoning": "I reviewed the purpose and usage of LangChain and suggested a plan for installation and testing", 		"plan": " Download and install LangChain. \n- Run code examples to test it out.",
		"criticism": "I should have provided more specific detailed steps on how to download and install LangChain",
		"speak": "We need to download and install LangChain before we can use it. Do you have any experience with this?",
    },
    "command":{
        "name": "browse website",
        "args": {
            "url": "https: //langchain.org/docs/getting-started/",
            "question": "How to download and install LangChain?"
        }
    }

System: Command browse_website returned: Error: HTTPSConnectionPool(host='langchain. org', port=443): Max retries exceeded with url: /docs/getting-started (Caused by NewConnectionError('<urllib3. connection. HTTPSConnection object at 0x7fbbb0243ee0>: Failed to establish a connection [Error 2] name or service not known'))

 User: GENERATE NEXT COMMAND JSON
```

AgentGPT 会将报错的信息 catch 组装一下，继续发给 ChatGPT，然后得到的回复是`- Conduct a Google search to find out how to download and install LangChain.`，意思是需要使用 Google 搜索，其中返回需要执行的命令：

```text
"command":{
    "name": "google",
    "args":{
        "input": "how to download and install LangChain"
    }
}
```

下一步就走到 google_search 的函数。再纠正自己的错误，循环最终解决问题。

### 整体循环

在 AgentGPT 是一步一步确认的，整个逻辑是在一个大循环中：

```python
# Interaction Loop
while True:
    # 发送数据给 AI，然后获得回应
    assistant_reply = chat.chat_with_ai(
            prompt,
            user_input,
            full_message_history,
            memory,
            cfg.fast_token_limit)
    # 解析以获得需要执行的命令和参数（由于返回的 JSON 需要校准，所以这里会有比较多的处理）
    # 执行命令
    # 将每次的问题和回复记录 memory 中
    # ...
```

其中`chat_with_ai`的参数：

1. prompt 是解释规则给 AI 的提示，主要有几个信息：
   1.  AI 的名字，每次对话自己设置的
   2.  描述 AI 需要帮你实现的事情
   3.  输入几个需要实现的目标（一般设置 5 个）
2. user_input 是用户输入
3. full_message_history 是用户和 AI 之间发送的所有消息的列表
4. memory 是包含永久记忆的内存对象
5. fast_token_limit 是 API 调用中允许的最大 token 数

### memory 的作用

AgentGPT 支持三种向量存储：pinecone，redis，local，默认使用 LocalCache。如下：

```python
if cfg.memory_backend == "pinecone":
    if not PineconeMemory:
        print("Error: Pinecone is not installed. Please install pinecone"
                " to use Pinecone as a memory backend.")
    else:
        memory = PineconeMemory(cfg)
        if init:
            memory.clear()
elif cfg.memory_backend == "redis":
    if not RedisMemory:
        print("Error: Redis is not installed. Please install redis-py to"
                " use Redis as a memory backend.")
    else:
        memory = RedisMemory(cfg)

if memory is None:
    memory = LocalCache(cfg)
    if init:
        memory.clear()
```

主要是控制 token 的数量，通过存储对话的上下文，找到全部历史信息中的最近 10 个关键上下文继续组成下一次上下文，比如 LocalCache 就是通过 ada 算法实现 topk 搜索：

```python
def get_relevant(self, text: str, k: int) -> List[Any]:
    embedding = get_ada_embedding(text)
    scores = np.dot(self.data.embeddings, embedding)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return [self.data.texts[i] for i in top_k_indices]
```

AgentGPT 有点像「套娃」，通过你输入的问题，不断去 ChatGPT 上查找解决方案，然后执行各个命令，如果命令返回不对，继续指定下一步计划，直到你得到结果为止。
专业一点的话来说，这叫模型堆叠，即模型「一路向下」，去套用别的模型来拆解并解决任务。
除了 AgentGPT、BabyAGI 这些，还有 ViperGPT、SayCan 和 ToolKit 等最新工具，以及前面微软发布的 VisualGPT 和 HugginGPT，都差不多是这个思想。

