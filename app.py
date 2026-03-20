import streamlit as st
import cv2
import imutils
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import InputLayer
from utils.image_processor import locate_puzzle, extract_digit
from utils.sudoku import Sudoku

# إعدادات الصفحة
st.set_page_config(page_title="محلل السودوكو الذكي", page_icon="🧩", layout="centered")

# ==========================================
# 🛡️ حل مشكلة التوافقية (Compatibility Patch)
# ==========================================
# نعدّل __init__ الأصلي لـ InputLayer مباشرة (مرة واحدة فقط)
# هذا يضمن أن أي كود داخلي في Keras يستخدم InputLayer سيمر بالتعديل
if not getattr(InputLayer, '_compatibility_patched', False):
    _original_input_layer_init = InputLayer.__init__

    def _fixed_input_layer_init(self, *args, **kwargs):
        # ✅ الإصلاح 1: تحويل batch_shape → batch_input_shape
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        # ✅ الإصلاح 2: حذف optional (غير موجود في الإصدارات القديمة)
        kwargs.pop('optional', None)
        _original_input_layer_init(self, *args, **kwargs)

    InputLayer.__init__ = _fixed_input_layer_init
    InputLayer._compatibility_patched = True  # منع التكرار عند إعادة التشغيل

# ==========================================
# ⚙️ إعدادات المحرك (القائمة الجانبية)
# ==========================================
st.sidebar.title("⚙️ إعدادات المحرك")
model_dir = "trained_model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

available_models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
if not available_models:
    st.sidebar.error(f"⚠️ لا يوجد ملفات .h5 في مجلد {model_dir}")
    st.stop()

selected_model_name = st.sidebar.selectbox(
    "🧠 اختر نموذج الذكاء الاصطناعي:",
    available_models
)
model_path = os.path.join(model_dir, selected_model_name)

@st.cache_resource
def load_ai_model(path):
    """تحميل النموذج - الباتش أعلاه يتكفل بالتوافقية تلقائياً"""
    return load_model(path, compile=False)

try:
    model = load_ai_model(model_path)
    st.sidebar.success(f"✅ النموذج النشط: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل الموديل: {e}")
    st.stop()

# ==========================================
# 🧩 الواجهة الرئيسية
# ==========================================
st.title("🧩 محلل السودوكو الذكي")
st.write("ارفع صورة اللغز، اختر الموديل، وصحح الأرقام لترى الحل مطبوعاً!")

if "board" not in st.session_state:
    st.session_state.board = None
    st.session_state.puzzle_image = None
    st.session_state.cell_locs = None

uploaded_file = st.file_uploader("رفع صورة السودوكو", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="الصورة الأصلية", channels="BGR", use_container_width=True)

    if st.button("🔍 تحليل الصورة واستخراج الأرقام"):
        with st.spinner(f"جاري التحليل باستخدام {selected_model_name}..."):
            image_resized = imutils.resize(image, width=600)
            puzzleImage, warped = locate_puzzle(image_resized, debug=False)

            board = np.zeros((9, 9), dtype="int")
            stepX = warped.shape[1] // 9
            stepY = warped.shape[0] // 9
            cellLocs = []

            for y in range(9):
                row = []
                for x in range(9):
                    startX, startY = x * stepX, y * stepY
                    endX, endY = (x + 1) * stepX, (y + 1) * stepY
                    cell = warped[startY:endY, startX:endX]
                    digit = extract_digit(cell, debug=False)

                    if digit is not None:
                        roi = cv2.resize(digit, (28, 28))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        pred = model.predict(roi).argmax(axis=1)[0]
                        board[y, x] = pred
                        row.append(None)
                    else:
                        row.append((startX, startY, endX, endY))
                cellLocs.append(row)

            st.session_state.board = board.tolist()
            st.session_state.puzzle_image = puzzleImage
            st.session_state.cell_locs = cellLocs
            st.success("✅ تم استخراج البيانات!")

if st.session_state.board is not None:
    st.write("### 📝 راجع وصحح الأرقام في الجدول:")
    df = pd.DataFrame(st.session_state.board)
    edited_df = st.data_editor(df, use_container_width=True, hide_index=True)

    if st.button("🚀 عرض الحل النهائي"):
        with st.spinner("جاري الحل..."):
            try:
                final_board = edited_df.fillna(0).astype(int).values.tolist()
                puzzle = Sudoku(final_board, 9, 9)
                puzzle.solve()

                res_img = st.session_state.puzzle_image.copy()
                for r in range(9):
                    for c in range(9):
                        loc = st.session_state.cell_locs[r][c]
                        if loc is not None:
                            val = puzzle.board[r][c]
                            startX, startY, endX, endY = loc
                            textX = int((endX - startX) * 0.33) + startX
                            textY = int((endY - startY) * 0.75) + startY
                            cv2.putText(
                                res_img,
                                str(val),
                                (textX, textY),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (255, 0, 0),
                                2,
                            )

                st.image(
                    res_img,
                    caption="الحل النهائي",
                    channels="BGR",
                    use_container_width=True,
                )
                st.balloons()
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
