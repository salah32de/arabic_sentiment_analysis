import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="تحليل المشاعر العربية",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== CSS مخصص ====================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    .main { direction: rtl; font-family: 'Cairo', sans-serif; }
    .stTextArea textarea { direction: rtl; text-align: right; font-family: 'Cairo', sans-serif; }
    
    .result-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 25px; border-radius: 15px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 8px 25px rgba(17,153,142,0.3);
    }
    .result-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white; padding: 25px; border-radius: 15px;
        text-align: center; font-size: 28px; font-weight: bold;
        box-shadow: 0 8px 25px rgba(235,51,73,0.3);
    }
    
    .title {
        text-align: center; color: #1f1f1f; font-size: 42px;
        font-weight: 700; margin-bottom: 10px; font-family: 'Cairo', sans-serif;
    }
    .subtitle {
        text-align: center; color: #666; font-size: 18px;
        margin-bottom: 30px; font-family: 'Cairo', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== دالة تنظيف النص ====================
def clean_arabic_text_for_model(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # 1. إزالة الروابط
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. إزالة الإشارات @ و #
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 3. إزالة الإيموجي
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # 4. توحيد الترقيم المكرر
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    
    # 5. إزالة التشكيل
    text = re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)
    
    # 6. توحيد الأحرف العربية
    text = re.sub(r"[إأآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)
    
    # 7. إزالة الأحرف المكررة (هههه → ه)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    
    # 8. إزالة الرموز غير العربية (مع الاحتفاظ بالمسافات)
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    
    # 9. إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 10. تقصير النص الطويل
    if len(text) > 500:
        text = text[:250]
    
    return text

# ==================== تحميل النموذج ====================
@st.cache_resource(show_spinner=False)
def load_my_model():
    tokenizer = AutoTokenizer.from_pretrained("./full_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("./full_model")
    return tokenizer, model

# ==================== دالة التنبؤ ====================
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)[0]
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probs[predicted_class].item()
    
    # تسمية الفئات (حسب تدريبك)
    labels = {0: "سلبي 😞", 1: "إيجابي 😊"}
    sentiment = labels.get(predicted_class, "غير معروف")
    
    return sentiment, confidence, predicted_class

# ==================== الواجهة الرئيسية ====================
def main():
    # العنوان
    st.markdown('<h1 class="title">🎭 تحليل المشاعر العربية</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">نموذج ذكاء اصطناعي متخصص في تحليل مشاعر النصوص العربية</p>', unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("### ⚙️ معلومات")
        st.info("""
        **النموذج**: BERT Arabic  
        **الفئات**: إيجابي / سلبي  
        **الدقة**: حسب تدريب النموذج
        """)
        
        st.markdown("### 💡 أمثلة للتجربة")
        examples = [
            "هذا المنتج رائع جداً وأنا سعيد بالشراء",
            "سوء الخدمة والتأخير المستمر أزعجني كثيراً",
            "الطقس معتدل والأمور طبيعية اليوم"
        ]
        
        for i, ex in enumerate(examples, 1):
            if st.button(f"مثال {i}", key=f"ex_{i}"):
                st.session_state.user_input = ex
    
    # تحميل النموذج
    with st.spinner("⏳ جاري تحميل النموذج..."):
        tokenizer, model = load_my_model()
    
    # حقل الإدخال
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    
    user_input = st.text_area(
        "📝 أدخل النص العربي:",
        value=st.session_state.user_input,
        height=120,
        placeholder="اكتب هنا مثال: هذا المنتج ممتاز وأنصح به..."
    )
    
    # زر التحليل
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("🔍 تحليل المشاعر", use_container_width=True)
    
    # التحليل والعرض
    if analyze:
        if not user_input.strip():
            st.warning("⚠️ الرجاء إدخال نص للتحليل")
        else:
            with st.spinner("⏳ جاري تحليل النص..."):
                # تنظيف
                cleaned = clean_arabic_text_for_model(user_input)
                
                if len(cleaned) < 2:
                    st.error("❌ النص غير واضح بعد التنظيف. حاول نصاً أطول.")
                else:
                    # تنبؤ
                    sentiment, confidence, cls = predict_sentiment(cleaned, tokenizer, model)
                    
                    # عرض النتيجة
                    css_class = "result-positive" if cls == 1 else "result-negative"
                    
                    st.markdown("---")
                    st.markdown(f'<div class="{css_class}">{sentiment}</div>', unsafe_allow_html=True)
                    
                    # شريط الثقة
                    st.markdown("### 📊 درجة الثقة")
                    st.progress(confidence)
                    st.markdown(f"<center><b>{confidence*100:.1f}%</b></center>", unsafe_allow_html=True)
                    
                    # تفاصيل إضافية
                    with st.expander("🔍 تفاصيل أكثر"):
                        st.write(f"**النص الأصلي:** {len(user_input)} حرف")
                        st.write(f"**النص المنظف:** {len(cleaned)} حرف")
                        st.write(f"**النص المعالج:** `{cleaned}`")
    
    # التذييل
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 14px; font-family: Cairo;">
            تم التطوير باستخدام 🤗 Transformers & Streamlit
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
