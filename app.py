"""
====================================================================
 โปรแกรมจำแนกโรค Covid-19 จากภาพ X-ray ด้วย Streamlit
====================================================================
คำอธิบายสำคัญเกี่ยวกับโมเดลของท่าน (สำคัญ อ่านก่อนใช้งาน):

จากการตรวจสอบไฟล์ .pkcls ทั้ง 3 ไฟล์ (W10_NN_model, W10_SVM_model,
W10_tree_model) พบว่า:

  1) โมเดลเหล่านี้ "ไม่ใช่" โมเดล scikit-learn ธรรมดา แต่เป็นโมเดลที่
     ฝึกและบันทึกออกมาจากโปรแกรม "Orange Data Mining" (บันทึกด้วย
     Orange.data.Model / joblib) ดังนั้นการโหลดไฟล์ .pkcls ต้อง
     ติดตั้งไลบรารี "Orange3" ไว้ในเครื่อง/เซิร์ฟเวอร์ด้วยเสมอ
     (ไม่สามารถโหลดด้วย scikit-learn ตรง ๆ ได้)

  2) ตัวแปรต้น (features) ของโมเดล "ไม่ใช่" ค่าตัวเลข/ข้อความที่คน
     กรอกเอง แต่เป็นเวกเตอร์ตัวเลข 2,048 ค่า ชื่อ n0, n1, ..., n2047
     ซึ่งเป็น "Image Embedding" (ลักษณะเด่นของภาพ) ที่สกัดออกมาจาก
     ภาพ X-ray ด้วยโมเดล Inception-v3 ผ่านวิดเจ็ต "Image Embedding"
     ของ Orange (ยืนยันได้จากชื่อคอลัมน์ n0...n2047 ซึ่งตรงกับรูปแบบ
     ที่วิดเจ็ตนี้สร้างขึ้นเป๊ะ ๆ)

  3) ตัวแปรผล (target) คือคอลัมน์ "category" มี 3 ค่า:
     'covid', 'normal', 'pneumonia'

ดังนั้น แอปนี้จึงถูกออกแบบให้ผู้ใช้ "อัปโหลดภาพ X-ray" แทนการกรอกตัวเลข
2,048 ช่องเอง โดยแอปจะเรียกใช้ตัวสกัดภาพ (Image Embedder) ของ Orange
ให้อัตโนมัติ เพื่อแปลงภาพเป็นเวกเตอร์ 2,048 มิติ ก่อนส่งเข้าโมเดล

*** ข้อควรระวัง: ขั้นตอนสกัดภาพ (ImageEmbedder) ต้องเชื่อมต่ออินเทอร์เน็ต
ไปยังเซิร์ฟเวอร์ของ Orange (https://api.garaza.io) เสมอ ถ้าเครื่องที่
รันแอปนี้ไม่มีอินเทอร์เน็ต หรือเซิร์ฟเวอร์ดังกล่าวล่ม การสกัดภาพจะล้มเหลว ***
====================================================================
"""

import streamlit as st
import numpy as np
import joblib
import tempfile
import os
from PIL import Image

# นำเข้าโมดูลของ Orange (จำเป็นสำหรับโหลดและรันโมเดล .pkcls)
from Orange.data import Table, Domain
from Orange.base import Model

# ------------------------------------------------------------------
# 1) ตั้งค่าหน้าเว็บ และหัวข้อแอป
# ------------------------------------------------------------------
st.set_page_config(page_title="Covid-19 X-ray Classifier", layout="centered")
st.title("โปรเเกรมจำเเนกโรค Covid-19 จากภาพ X-ray")

st.markdown(
    """
    อัปโหลดไฟล์โมเดล (.pkcls) และภาพ X-ray ปอด แล้วกดปุ่ม **"ทำนายผล"**
    เพื่อให้โมเดลทำนายว่าภาพนี้เป็น **Covid-19 / ปกติ (Normal) /
    ปอดอักเสบ (Pneumonia)**
    """
)

# แปลชื่อคลาสภาษาอังกฤษ -> ภาษาไทย เพื่อแสดงผลให้อ่านง่าย
CLASS_LABEL_TH = {
    "covid": "โควิด-19 (Covid-19)",
    "normal": "ปกติ (Normal)",
    "pneumonia": "ปอดอักเสบ (Pneumonia)",
}

st.divider()

# ------------------------------------------------------------------
# 2) ส่วนเลือกไฟล์โมเดล (.pkcls) โดยผู้ใช้เอง
# ------------------------------------------------------------------
st.subheader("1) เลือกไฟล์โมเดล (.pkcls)")

st.info(
    "กรุณาดาวน์โหลดไฟล์โมเดล (เช่น W10_NN_model.pkcls, "
    "W10_SVM_model.pkcls หรือ W10_tree_model.pkcls) จาก Google Drive "
    "มาไว้ในเครื่องก่อน แล้วอัปโหลดไฟล์ผ่านช่องด้านล่างนี้"
)

uploaded_model_file = st.file_uploader(
    "อัปโหลดไฟล์โมเดล (.pkcls)",
    type=["pkcls"],
    help="โมเดลที่ฝึกและบันทึกไว้จากโปรแกรม Orange Data Mining",
)

# เก็บโมเดลที่โหลดสำเร็จไว้ใน session_state กันหายเวลากดปุ่มอื่น
if "model" not in st.session_state:
    st.session_state["model"] = None
if "model_name" not in st.session_state:
    st.session_state["model_name"] = None

if uploaded_model_file is not None:
    try:
        # joblib.load ต้องการ path ไฟล์ จึงบันทึกไฟล์ที่อัปโหลดลง
        # ไฟล์ชั่วคราวก่อน แล้วค่อยโหลด
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkcls") as tmp_file:
            tmp_file.write(uploaded_model_file.getbuffer())
            tmp_model_path = tmp_file.name

        model = joblib.load(tmp_model_path)
        st.session_state["model"] = model
        st.session_state["model_name"] = uploaded_model_file.name

        os.remove(tmp_model_path)

        st.success(f"โหลดโมเดลสำเร็จ: {uploaded_model_file.name}")
        st.caption(
            f"ประเภทโมเดล: `{type(model).__name__}` | "
            f"จำนวนตัวแปรต้น: {len(model.domain.attributes)} | "
            f"คลาสเป้าหมาย: {', '.join(model.domain.class_var.values)}"
        )
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดขณะโหลดโมเดล: {e}")
        st.info(
            "หากขึ้น error เกี่ยวกับโมดูล 'Orange' กรุณาตรวจสอบว่าได้ติดตั้ง "
            "Orange3 (และ Orange3-ImageAnalytics, PyQt5) ตามไฟล์ requirements.txt แล้ว"
        )
        st.session_state["model"] = None

st.divider()

# ------------------------------------------------------------------
# 3) ส่วนอัปโหลดภาพ X-ray (แทนการกรอกตัวเลข 2,048 ค่าด้วยมือ)
# ------------------------------------------------------------------
st.subheader("2) อัปโหลดภาพ X-ray")

uploaded_image = st.file_uploader(
    "อัปโหลดภาพ X-ray (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="ภาพ X-ray ที่อัปโหลด", width=300)

st.divider()

# ------------------------------------------------------------------
# 4) ปุ่มทำนายผล
# ------------------------------------------------------------------
st.subheader("3) ทำนายผล")

predict_button = st.button("ทำนายผล")

if predict_button:
    # ---- 4.1 ตรวจสอบความพร้อมก่อนทำนาย ----
    if st.session_state["model"] is None:
        st.warning("กรุณาอัปโหลดไฟล์โมเดล (.pkcls) ก่อน")
    elif uploaded_image is None:
        st.warning("กรุณาอัปโหลดภาพ X-ray ก่อน")
    else:
        model = st.session_state["model"]
        try:
            # ------------------------------------------------
            # 4.2) บันทึกภาพที่อัปโหลดลงไฟล์ชั่วคราว
            #      (ImageEmbedder ต้องการ path ของไฟล์ภาพ)
            # ------------------------------------------------
            suffix = os.path.splitext(uploaded_image.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_img:
                tmp_img.write(uploaded_image.getbuffer())
                tmp_img_path = tmp_img.name

            # ------------------------------------------------
            # 4.3) สกัดภาพเป็นเวกเตอร์ 2,048 มิติด้วย Inception-v3
            #      *** ต้องใช้ model เดียวกับตอนฝึก (inception-v3) ***
            #      *** ต้องเชื่อมต่ออินเทอร์เน็ตไปยัง Orange server ***
            # ------------------------------------------------
            with st.spinner("กำลังสกัดคุณลักษณะของภาพ (Image Embedding)..."):
                # นำเข้าตรงนี้เพื่อให้แอปยังเปิดได้แม้ยังไม่ได้ติดตั้ง
                # Orange3-ImageAnalytics ครบถ้วน (จะ error เฉพาะตอนกดทำนาย)
                from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

                with ImageEmbedder(model="inception-v3") as embedder:
                    embeddings = embedder([tmp_img_path])

            os.remove(tmp_img_path)

            if embeddings is None or embeddings[0] is None:
                st.error(
                    "สกัดคุณลักษณะของภาพไม่สำเร็จ (embedding = None) "
                    "สาเหตุที่พบบ่อยที่สุดคือ **ไม่มีการเชื่อมต่ออินเทอร์เน็ต** "
                    "ไปยังเซิร์ฟเวอร์ของ Orange (https://api.garaza.io) "
                    "กรุณาตรวจสอบอินเทอร์เน็ตของเครื่อง/เซิร์ฟเวอร์ที่รันแอปนี้ "
                    "แล้วลองใหม่อีกครั้ง"
                )
            else:
                # ------------------------------------------------
                # 4.4) สร้าง Orange Table จากเวกเตอร์ภาพ แล้วส่งเข้าโมเดล
                #      ใช้ domain เดียวกับตอนฝึก (n0..n2047) เพื่อให้
                #      ลำดับ/ชื่อฟีเจอร์ตรงกันทุกประการ
                # ------------------------------------------------
                X = np.array(embeddings, dtype=float)  # shape (1, 2048)

                # domain สำหรับข้อมูลนำเข้า (ไม่มี class เพราะยังไม่รู้คำตอบ)
                input_domain = Domain(model.domain.attributes)
                input_table = Table.from_numpy(input_domain, X)

                # ทำนายค่า + ความน่าจะเป็นของแต่ละคลาส
                predicted_idx, probs = model(input_table, Model.ValueProbs)

                class_values = model.domain.class_var.values
                predicted_class = class_values[int(predicted_idx[0])]
                predicted_prob = float(np.max(probs[0])) * 100

                # ------------------------------------------------
                # 4.5) แสดงผลลัพธ์ให้อ่านง่าย
                # ------------------------------------------------
                label_th = CLASS_LABEL_TH.get(predicted_class, predicted_class)

                if predicted_class == "covid":
                    st.error(
                        f"ผลการทำนาย: **{label_th}** "
                        f"(ความมั่นใจ {predicted_prob:.2f}%)"
                    )
                elif predicted_class == "pneumonia":
                    st.warning(
                        f"ผลการทำนาย: **{label_th}** "
                        f"(ความมั่นใจ {predicted_prob:.2f}%)"
                    )
                else:
                    st.success(
                        f"ผลการทำนาย: **{label_th}** "
                        f"(ความมั่นใจ {predicted_prob:.2f}%)"
                    )

                # แสดงความน่าจะเป็นของทุกคลาสเป็นตาราง
                st.write("ความน่าจะเป็นของแต่ละคลาส:")
                prob_dict = {
                    CLASS_LABEL_TH.get(cls, cls): f"{p * 100:.2f}%"
                    for cls, p in zip(class_values, probs[0])
                }
                st.table(prob_dict)

        except ModuleNotFoundError as e:
            st.error(
                f"ไม่พบไลบรารีที่จำเป็น: {e}\n\n"
                "กรุณาติดตั้ง Orange3, Orange3-ImageAnalytics และ PyQt5 "
                "ตามไฟล์ requirements.txt ก่อนใช้งาน"
            )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดขณะทำนายผล: {e}")
