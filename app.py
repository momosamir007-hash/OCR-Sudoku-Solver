import streamlit as st
import subprocess
import os

st.set_page_config(page_title="محلل السودوكو الذكي", page_icon="🧩")

st.title("🧩 محلل ألغاز السودوكو بالذكاء الاصطناعي")
st.write("قم برفع صورة للغز السودوكو وسيقوم البرنامج بقراءتها وحلها فوراً!")

# أداة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة السودوكو (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # حفظ الصورة مؤقتاً لكي يقرأها الكود الأصلي
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # عرض الصورة للمستخدم
    st.image("temp_image.jpg", caption="الصورة المرفوعة", use_container_width=True)
    
    if st.button("حل اللغز الآن 🚀"):
        with st.spinner("جاري تحليل الصورة وحل اللغز... الرجاء الانتظار"):
            # تشغيل كود الحل الأصلي وتخزين النتيجة
            process = subprocess.run(
                ["python", "solve_sudoku.py", "-m", "trained_model/digit_classifier.h5", "-i", "temp_image.jpg"],
                capture_output=True, text=True
            )
            
            # طباعة النتيجة النهائية
            st.success("تم الحل بنجاح!")
            st.code(process.stdout, language="text")
            
            # في حال وجود خطأ
            if process.stderr and "Error" in process.stderr:
                st.error("حدث خطأ أثناء التحليل:")
                st.code(process.stderr, language="text")
