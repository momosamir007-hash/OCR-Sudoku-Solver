import streamlit as st
import subprocess
import os
import sys  # أضفنا هذه المكتبة لضمان استخدام المسار الصحيح لبايثون

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
            
            # تشغيل كود الحل الأصلي باستخدام sys.executable لضمان قراءة مكتبة cv2
            process = subprocess.run(
                [sys.executable, "solve_sudoku.py", "-m", "trained_model/digit_classifier.h5", "-i", "temp_image.jpg"],
                capture_output=True, text=True
            )
            
            # التحقق مما إذا كانت العملية قد نجحت (الكود 0 يعني نجاح تام)
            if process.returncode == 0:
                st.success("تم الحل بنجاح!")
                st.code(process.stdout, language="text")
            else:
                # في حال فشل الكود، سنعرض رسالة الخطأ الدقيقة لمعرفة السبب
                st.error("حدث خطأ أثناء التحليل:")
                st.code(process.stderr, language="text")
                
                # عرض أي نصوص أخرى طبعها البرنامج قبل أن يتوقف (تساعدنا جداً في التتبع)
                if process.stdout:
                    st.warning("سجل العمليات (قد يساعد في معرفة أين توقف البرنامج):")
                    st.code(process.stdout, language="text")
