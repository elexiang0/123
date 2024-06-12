import streamlit as st
from litellm import completion

# Streamlit 应用程序界面
st.title('中英文翻译应用')

# 用户输入中文文本
input_text = st.text_input('请输入中文文本', '')

# 当用户输入文本后, 进行翻译并显示结果
if st.button('翻译为英文'):
    if input_text:
        # 使用 LiteLLM调用 Deepseek模型进行翻译
        response = completion(
            model='deepseek/deepseek-chat',
            messages=[
                {
                    "content": "你是一个优秀的英文翻译, 请根据用户输入和英文读者的阅读习惯, 把内容原原本本翻译成对应的英文",
                    "role": "system"
                },
                {
                    "content": input_text,
                    "role": "user"
                }
            ]
        )
        translated_text = response['choices'][0]['message']['content']
        st.write('英文翻译结果:', translated_text)
    else:
        st.write('请输入中文文本后再进行翻译')
