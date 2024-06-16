import streamlit as st
from fastai.vision.all import *
from pathlib import Path

# 加载训练好的模型
model_path = Path("C:/Users/25436/Desktop/sdxx/path_to_save_model.pkl")
learn = load_learner(model_path)

# 紧急处理预案字典
emergency_measures = {
    'king cobra': '立即呼叫急救服务，保持镇静，减少活动。',
    'Krait': '立即呼叫急救服务，保持镇静，避免移动被咬部位。',
    'Pallas pit viper': '立即就医，保持冷静，尽量减少活动。',
    'Silver Ring Snake': '立即呼叫急救服务，保持镇静，尽量减少活动。',
    'Trimeresurus stejnegeri': '立即就医，保持冷静，尽量减少活动。'
}

# Streamlit 界面
st.title("蛇类图片分类器")
st.write("请上传一张蛇类图片，我们将帮你识别该蛇种并提供紧急处理预案。")

uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PILImage.create(uploaded_file)
    st.image(img, caption='上传的图片', use_column_width=True)
    st.write("")
    st.write("分类中...")

    pred, pred_idx, probs = learn.predict(img)
    st.write(f"预测类别: {pred}")
    st.write(f"置信度: {probs[pred_idx]:.4f}")

    st.write("紧急处理预案:")
    st.write(emergency_measures.get(pred, "未知的蛇种，请咨询专业人员。"))

# 运行应用程序
# 使用命令 `streamlit run app.py` 来启动应用程序
