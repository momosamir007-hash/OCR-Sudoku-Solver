import streamlit as st
import cv2
import imutils
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# استدعاء أدوات المشروع الأصلي مباشرة
from utils.image_processor import locate_puzzle, extract_digit
from utils.sudoku import Sudoku

st.set_page_config(page_title="محلل السودوكو الذكي", page_icon="🧩", layout="centered")

st.title("🧩 محلل السودوكو (مع التعديل اليدوي)")
st.write("الآن يمكنك مراجعة قراءة الذكاء الاصطناعي وتصحيحها قبل الحل!")

# تحميل الموديل مرة واحدة فقط لتسريع التطبيق
@st.cache_resource
def load_ai_model():
    return load_model("trained_model/digit_classifier.h5")

model = load_ai_model()

# مساحة لتخزين الشبكة في ذاكرة التطبيق
if "board" not in st.session_state:
    st.session_state.board = None

# أداة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة السودوكو (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # حفظ الصورة مؤقتاً
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image("temp_image.jpg", caption="الصورة المرفوعة", width=350)
    
    # زر استخراج الأرقام
    if st.button("🔍 استخراج الأرقام من الصورة"):
        with st.spinner("الذكاء الاصطناعي يقرأ الأرقام..."):
            image = cv2.imread("temp_image.jpg")
            image = imutils.resize(image, width=600)
            
            # البحث عن المربعات واستخراجها
            puzzleImage, warped = locate_puzzle(image, debug=False)
            board = np.zeros((9, 9), dtype='int')
            
            stepX = warped.shape[1] // 9
            stepY = warped.shape[0] // 9
            
            for y in range(9):
                for x in range(9):
                    startX = x * stepX
                    startY = y * stepY
                    endX = (x + 1) * stepX
                    endY = (y + 1) * stepY
                    
                    cell = warped[startY:endY, startX:endX]
                    digit = extract_digit(cell, debug=False)
                    
                    if digit is not None:
                        roi = cv2.resize(digit, (32, 32))
                        roi = roi.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        pred = model.predict(roi).argmax(axis=1)[0]
                        board[y, x] = pred
            
            # حفظ النتيجة في الذاكرة لعرضها
            st.session_state.board = board.tolist()
            st.success("تم استخراج الأرقام بنجاح! راجع الجدول أدناه.")

    # إذا تم استخراج الشبكة، نعرضها للتعديل
    if st.session_state.board is not None:
        st.write("---")
        st.write("### 📝 شبكة الأرقام (قابلة للتعديل):")
        st.info("💡 **نصيحة:** راجع الأرقام جيداً. إذا أخطأ الذكاء الاصطناعي في قراءة رقم، اضغط عليه وصححه. (رقم 0 يعني مربع فارغ).")
        
        # تحويل الشبكة لجدول تفاعلي
        df = pd.DataFrame(st.session_state.board)
        
        # عرض الجدول القابل للتعديل
        edited_df = st.data_editor(df, use_container_width=True, hide_index=True)
        
        # زر الحل النهائي
        if st.button("🚀 حل اللغز الآن"):
            with st.spinner("جاري الحل..."):
                try:
                    # أخذ الأرقام بعد تعديل المستخدم
                    final_board = edited_df.values.tolist()
                    puzzle = Sudoku(final_board, 9, 9)
                    puzzle.solve()
                    
                    st.success("🎉 تم حل اللغز بنجاح!")
                    st.write("### ✅ النتيجة النهائية:")
                    
                    # عرض النتيجة في جدول جميل
                    solved_df = pd.DataFrame(puzzle.board)
                    
                    # تلوين الجدول (اختياري لجمالية العرض)
                    st.dataframe(solved_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error("⚠️ تعذر حل اللغز! تأكد من أن الأرقام التي أدخلتها صحيحة ولا تخالف قواعد لعبة السودوكو.")
