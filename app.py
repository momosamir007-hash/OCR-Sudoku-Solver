import streamlit as st
import cv2
import imutils
import numpy as np
import pandas as pd
import os
import json
import shutil
import tempfile
import h5py
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import InputLayer
from utils.image_processor import locate_puzzle, extract_digit
from utils.sudoku import Sudoku

st.set_page_config(page_title="محلل السودوكو الذكي", page_icon="🧩", layout="centered")

# =====================================================
# 🛡️ محوّل التوافقية: Keras 3 ← → Keras 2
# =====================================================

def _convert_keras3_to_keras2(obj):
    """
    تحويل تنسيق Keras 3 التسلسلي إلى تنسيق Keras 2
    يعالج كل الطبقات: InputLayer, Conv2D, Dense, BatchNorm...
    """
    # ---- القوائم: ندخل في كل عنصر ----
    if isinstance(obj, list):
        return [_convert_keras3_to_keras2(item) for item in obj]

    # ---- أي شيء غير dict: نرجعه كما هو ----
    if not isinstance(obj, dict):
        return obj

    # ---- كشف كائن Keras 3 المسلسل ----
    # الشكل: {"module": "...", "class_name": "...", "config": {...}, "registered_name": ...}
    is_keras3_obj = (
        "module" in obj and
        "class_name" in obj and
        "registered_name" in obj and
        "config" in obj
    )

    if is_keras3_obj:
        class_name = obj["class_name"]
        inner_config = obj.get("config", {})

        # ✅ حالة خاصة: DTypePolicy → نرجع اسم النوع فقط كنص
        if class_name == "DTypePolicy":
            if isinstance(inner_config, dict):
                return inner_config.get("name", "float32")
            return "float32"

        # ✅ نحوّل الـ config الداخلي بشكل تراجعي
        fixed_inner = _convert_keras3_to_keras2(inner_config)

        # ✅ حالة خاصة: InputLayer
        if class_name == "InputLayer" and isinstance(fixed_inner, dict):
            if "batch_shape" in fixed_inner:
                fixed_inner["batch_input_shape"] = fixed_inner.pop("batch_shape")
            fixed_inner.pop("optional", None)

        # ✅ إرجاع بتنسيق Keras 2 (بدون module و registered_name)
        return {
            "class_name": class_name,
            "config": fixed_inner,
        }

    # ---- dict عادي: ندخل في كل القيم ----
    return {k: _convert_keras3_to_keras2(v) for k, v in obj.items()}

def fix_h5_for_keras2(original_path):
    """
    ينشئ نسخة مؤقتة من ملف .h5 مع config مُحوَّل لـ Keras 2
    يرجع مسار الملف المؤقت
    """
    # إنشاء ملف مؤقت
    temp_fd, temp_path = tempfile.mkstemp(suffix=".h5")
    os.close(temp_fd)
    shutil.copy2(original_path, temp_path)

    with h5py.File(temp_path, "r+") as f:
        # ---- تحويل model_config ----
        if "model_config" in f.attrs:
            raw = f.attrs["model_config"]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            config = json.loads(raw)
            fixed = _convert_keras3_to_keras2(config)
            f.attrs["model_config"] = json.dumps(fixed).encode("utf-8")

        # ---- حذف سمات Keras 3 الزائدة ----
        for attr_name in ["build_config", "compile_config"]:
            if attr_name in f.attrs:
                del f.attrs[attr_name]

    return temp_path

# =====================================================
# 🛡️ باتش أمان إضافي لـ InputLayer
# =====================================================

if not getattr(InputLayer, "_patched", False):
    _orig_init = InputLayer.__init__

    def _safe_init(self, *args, **kwargs):
        if "batch_shape" in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        kwargs.pop("optional", None)
        _orig_init(self, *args, **kwargs)

    InputLayer.__init__ = _safe_init
    InputLayer._patched = True

# =====================================================
# ⚙️ إعدادات المحرك
# =====================================================

st.sidebar.title("⚙️ إعدادات المحرك")
model_dir = "trained_model"
os.makedirs(model_dir, exist_ok=True)

available_models = [f for f in os.listdir(model_dir) if f.endswith(".h5")]
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
    """
    محاولة 1: تحميل عادي (للموديلات القديمة)
    محاولة 2: تحويل h5 من Keras3→Keras2 ثم تحميل
    """
    # ---- المحاولة 1: تحميل مباشر ----
    try:
        return load_model(path, compile=False)
    except Exception:
        pass

    # ---- المحاولة 2: تحويل التنسيق ثم تحميل ----
    temp_path = None
    try:
        temp_path = fix_h5_for_keras2(path)
        model = load_model(temp_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(
            f"فشل تحميل الموديل حتى بعد تحويل التنسيق.\n"
            f"تأكد أن إصدار TensorFlow متوافق.\n"
            f"التفاصيل: {e}"
        )
    finally:
        # تنظيف الملف المؤقت
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

try:
    model = load_ai_model(model_path)
    st.sidebar.success(f"✅ النموذج النشط: {selected_model_name}")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في تحميل الموديل:\n{e}")
    st.stop()

# =====================================================
# 🧩 الواجهة الرئيسية
# =====================================================

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
                            sX, sY, eX, eY = loc
                            tX = int((eX - sX) * 0.33) + sX
                            tY = int((eY - sY) * 0.75) + sY
                            cv2.putText(
                                res_img,
                                str(val),
                                (tX, tY),
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
