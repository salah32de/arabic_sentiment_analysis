import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="arabic sentiment analysis", page_icon="🤖")

st.title("arabic sentiment")
st.markdown("model train with arabic dataset (pos/neg)")

@st.cache_resource
def load_my_model():
    # تحميل التوكينايزر والموديل
    tokenizer = AutoTokenizer.from_pretrained("./full_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("./full_model")
    
    return tokenizer, model

tokenizer, model = load_my_model()

# دالة للتنبؤ
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # افترض أن 0 = سلبي، 1 = إيجابي (يمكنك تبديل حسب تدريبك)
    sentiment = "positive 😊" if predicted_class == 1 else "negative 😞"
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    return sentiment, confidence

# واجهة المستخدم
arabic_text = st.text_area("📝 أدخل النص العربي:", height=150)

if st.button("🔍 حلل المشاعر"):
    if arabic_text.strip():
        sentiment, conf = predict_sentiment(arabic_text)
        if "إيجابي" in sentiment:
            st.success(f"{sentiment} (الثقة: {conf:.2%})")
        else:
            st.error(f"{sentiment} (الثقة: {conf:.2%})")
    else:
        st.warning("Enter the text") 