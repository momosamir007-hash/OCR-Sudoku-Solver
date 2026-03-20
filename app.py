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

st.title("🧩 محلل السودوكو الذكي التفاعلي")
st.write("قم برفع الصورة، راجع الأرقام وصححها إن لزم الأمر، وشاهد الحل مطبوعاً على الصورة!")

# تحميل الموديل مرة واحدة فقط لتسريع التطبيق
@st.cache_resource
def load_ai_model():
    return load_model("trained_model/digit_classifier.h5")

model = load_ai_model()

# مساحة لتخزين المتغيرات في ذاكرة التطبيق
if "board" not in st.session_state:
    st.session_state.board = None
    st.session_state.puzzle_image = None
    st.session_state.cell_locs = None

# أداة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة السودوكو (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # حفظ الصورة مؤقتاً
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image("temp_image.jpg", caption="الصورة الأصلية المرفوعة", use_container_width=True)
    
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
            
            cellLocs = [] # لحفظ أماكن المربعات لرسم الحل عليها لاحقاً
            
            for y in range(9):
                row = []
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
                        row.append(None) # مربع ممتلئ أساساً
                    else:
                        row.append((startX, startY, endX, endY)) # حفظ إحداثيات المربع الفارغ
                cellLocs.append(row)
            
            # حفظ النتيجة والصورة والإحداثيات في الذاكرة
            st.session_state.board = board.tolist()
            st.session_state.puzzle_image = puzzleImage.copy()
            st.session_state.cell_locs = cellLocs
            
            st.success("تم استخراج الأرقام بنجاح! راجع الجدول أدناه.")

    # إذا تم استخراج الشبكة، نعرضها للتعديل
    if st.session_state.board is not None:
        st.write("---")
        st.write("### 📝 شبكة الأرقام (قابلة للتعديل):")
        st.info("💡 **نصيحة:** راجع الأرقام. إذا أخطأ الذكاء الاصطناعي في قراءة أي رقم من الصورة، اضغط عليه وصححه هنا (0 يعني فراغ).")
        
        # تحويل الشبكة لجدول تفاعلي
        df = pd.DataFrame(st.session_state.board)
        edited_df = st.data_editor(df, use_container_width=True, hide_index=True)
        
        # زر الحل النهائي
        if st.button("🚀 حل اللغز الآن"):
            with st.spinner("جاري حل اللغز ورسم النتيجة..."):
                try:
                    # 💡 الإصلاح هنا: إجبار الجدول على أن يكون أرقاماً (int) لتجنب خطأ الأصفار
                    final_board = edited_df.fillna(0).astype(int).values.tolist()
                    
                    puzzle = Sudoku(final_board, 9, 9)
                    puzzle.solve()
                    
                    st.success("🎉 تم حل اللغز بنجاح!")
                    
                    # رسم الأرقام على الصورة
                    output_image = st.session_state.puzzle_image.copy()
                    
                    for (cellRow, boardRow) in zip(st.session_state.cell_locs, puzzle.board):
                        for (cell, digit) in zip(cellRow, boardRow):
                            if cell is None:
                                continue # تخطي المربعات التي كانت ممتلئة في الأصل
                            
                            # أمان إضافي: لا تقم برسم الرقم إذا كان 0
                            if digit == 0:
                                continue
                            
                            startX, startY, endX, endY = cell
                            
                            # حساب مكان وضع الرقم (توسيط تقريبي)
                            testX = int((endX - startX) * 0.33) + startX
                            testY = int((endY - startY) * -0.2) + endY
                            
                            # رسم الرقم باللون الأزرق على الصورة
                            cv2.putText(output_image, str(digit), (testX, testY),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    st.write("### ✅ الصورة النهائية بعد الحل:")
                    # عرض الصورة في Streamlit
                    st.image(output_image, channels="BGR", use_container_width=True)
                    
                    # إبقاء الجدول لعرض النتيجة كنص أيضاً
                    with st.expander("عرض النتيجة كجدول أرقام"):
                        solved_df = pd.DataFrame(puzzle.board)
                        st.dataframe(solved_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error("⚠️ تعذر حل اللغز! تأكد من أن الأرقام المدخلة في الجدول صحيحة ولا تخالف قواعد اللعبة.")
