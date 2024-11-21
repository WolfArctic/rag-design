# 导入Gradio库，用于创建交互式Web应用程序
import gradio as gr

# 从llm模块中导入MyLLM类，这是自定义的大型语言模型接口
from llm import MyLLM

# 定义一个包含两个LLM模型名称的列表，供用户选择
LLM_MODELS = ["gpt-3.5-turbo", "gpt-4o"]

# 实例化MyLLM类，用于后续的模型调用和处理
llm = MyLLM()

# 定义submit函数，用于处理用户提交的查询
# query => 是用户输入
# chat_history => 聊天的历史记录
def submit(query, chat_history):
    print('query:',query)
    print('chat_history:',chat_history)
    f""" 搜索知识库 """
    # 如果查询为空字符串，返回空字符串和当前的聊天记录
    if query == '':
        return '', chat_history
    # 如果查询不为空，将查询添加到聊天记录中，并返回更新后的聊天记录
    chat_history.append([query, None]) # 还没有使用到模型的回复
    print('chat_history####:',chat_history)
    print('('', chat_history):',('', chat_history))
    return '', chat_history

# 定义 change_collection 函数，用于在更改知识库时清除模型历史记录
def change_collection():
    llm.clear_history()

# 定义load_history函数，用于加载模型的历史消息
def load_history():
    # 将历史消息格式化为对话形式，每两条消息为一对
    history = llm.get_history_message()
    return [
        [history[i].content, history[i + 1].content]
        if i + 1 < len(history) else
        [history[i].content, None]
        for i in range(0, len(history), 2)
        ]

# 定义llm_reply函数，用于生成模型回复
def llm_reply(collection, chat_history, model, max_length=256, temperature=1):
    question = chat_history[-1][0]
    print('question:',question)
    # 使用流式生成方法从模型中获取回复
    response = llm.stream(question, collection, model=model, max_length=max_length, temperature=temperature)
    print('response:',response)
    # print('responselist:',list(response))
    chat_history[-1][1] = ""
    print('chat_history:',chat_history)

    # 逐块处理模型生成的回复
    for chunk in response:
        print('chunk:',chunk)
        if 'context' in chunk:
            # 如果块中包含上下文信息，则打印出来
            for doc in chunk['context']:
                print('doc:',doc)
        if 'answer' in chunk:
            # 如果块中包含答案，则将其追加到聊天记录的最后一个条目中
            chunk_content = chunk['answer']
            print('chunk_content:',chunk_content)
            if chunk_content is not None:
                chat_history[-1][1] += chunk_content
                # 返回更新后的聊天记录
                yield chat_history
    print('chat_history:', chat_history)

# 创建一个Gradio Blocks应用，设置fill_height为True
with gr.Blocks(fill_height=True) as demo:
    # 在应用中添加一个HTML元素，显示标题
    gr.HTML("""<h1 align="center">Chat With AI</h1>""")

    # 创建一个新的行布局
    with gr.Row():
        # 创建一个占比为 4 的列布局
        with gr.Column(scale=4):
            # 创建一个下拉菜单，用于选择LLM模型
            model = gr.Dropdown(
                choices=LLM_MODELS,
                value=LLM_MODELS[1],
                label="LLM Model",
                interactive=True,
                scale=1
            )
            # 创建一个聊天机器人界面
            chatbot = gr.Chatbot(show_label=False, scale=3, show_copy_button=True)

        # 创建一个占比为 1 的列布局，显示进度
        with gr.Column(scale=1, show_progress=True) as column_config:
            # 创建一个滑块，用于设置生成回复的最大长度
            max_length = gr.Slider(1, 4095, value=256, step=1.0, label="Maximum length", interactive=True)
            # 创建一个滑块，用于设置生成回复的温度
            temperature = gr.Slider(0, 2, value=1, step=0.01, label="Temperature", interactive=True)
            # 创建一个按钮，用于清除聊天记录
            clear = gr.Button("清除")
            # 创建一个下拉菜单，用于选择知识库
            collection = gr.Dropdown(choices=llm.load_knowledge(), label="知识库")
            # 创建一个文件上传控件，支持多种文件类型
            file = gr.File(label="上传文件", file_types=['doc', 'docx', 'csv', 'txt', 'pdf', 'md'])

    # 创建一个文本框，用于用户输入
    user_input = gr.Textbox(placeholder="Input...", show_label=False)
    # 创建一个按钮，用于提交用户输入
    user_submit = gr.Button("提交")

    # 绑定 clear 按钮的点击事件，清除模型历史记录，并更新聊天机器人界面
    clear.click(fn=llm.clear_history, inputs=None, outputs=[chatbot])

    # 当我们询问模型的时候,按的回车键,会触发 => submit 事件
    user_input.submit(fn=submit,
                      inputs=[user_input, chatbot], # 一个是用户输入 一个当前的聊天记录
                      outputs=[user_input, chatbot] # 一个是为了清空用户输入的文本框 另一个是更新后的聊天记录，将新的用户查询添加到聊天记录中。
                      ).then( # 执行代码的时候有先后顺序
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
                # collection: 用户选择的知识库。
                # chatbot: 当前的聊天记录（已经包含用户的新查询）。
                # model: 用户选择的 LLM 模型。
                # max_length: 用户设置的生成回复的最大长度。
                # temperature: 用户设置的生成回复的温度。
        outputs=[chatbot]
        # 更新后的聊天记录，将模型生成的回复添加到聊天记录中。
    )
    # 绑定用户输入文本框的提交事件，
    # 先调用submit函数，
    # 然后调用llm_reply函数，
    # 并更新聊天机器人界面
    user_submit.click(fn=submit,
                      inputs=[user_input, chatbot],
                      outputs=[user_input, chatbot]
                      ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )
    # 绑定提交按钮的点击事件，先调用submit函数，
    # 然后调用llm_reply函数，并更新聊天机器人界面

    # 绑定文件上传控件的上传事件，调用upload_knowledge函数，并更新文件控件和知识库下拉菜单
    file.upload(fn=llm.upload_knowledge, inputs=[file], outputs=[file, collection])

    # 绑定知识库下拉菜单的更改事件，调用clear_history函数，并更新聊天机器人界面 也就是换一个知识库就清空当前的页面
    collection.change(fn=llm.clear_history, inputs=None, outputs=[chatbot])

    # 绑定应用加载事件，调用clear_history函数，并更新聊天机器人界面
    demo.load(fn=llm.clear_history, inputs=None, outputs=[chatbot])

# 启动 Gradio 应用
demo.launch(share=True)
