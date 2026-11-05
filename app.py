import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Laptop Price Predictor")


@st.cache_resource
def load_model():
    return joblib.load("laptop_model.joblib")


bundle = load_model()
model, columns, options = bundle["model"], bundle["columns"], bundle["options"]

st.title("Laptop Price Predictor")
st.write("Choose the laptop's specifications and click **Predict**.")

col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Brand", options["Company"], index=options["Company"].index("Dell"))
    type_name = st.selectbox("Type", options["TypeName"], index=options["TypeName"].index("Notebook"))
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
    touchscreen = st.checkbox("Touchscreen")
    ips = st.checkbox("IPS panel")
    os_name = st.selectbox("Operating system", options["OS"], index=options["OS"].index("Windows"))
with col2:
    screen_size = st.number_input("Screen size (inches)", min_value=10.0, max_value=18.4, value=15.6, step=0.1)
    resolution = st.selectbox(
        "Screen resolution",
        ["1366x768", "1600x900", "1920x1080", "2560x1440", "2880x1800", "3200x1800", "3840x2160"],
        index=2,
    )
    cpu_brand = st.selectbox("Processor", options["Cpu_brand"], index=options["Cpu_brand"].index("Intel Core i5"))
    cpu_ghz = st.number_input("Processor speed (GHz)", min_value=0.9, max_value=3.6, value=2.5, step=0.1)
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024], index=2)
    hdd = st.selectbox("HDD (GB)", [0, 500, 1000, 2000], index=0)
    gpu_brand = st.selectbox("Graphics card", options["Gpu_brand"], index=options["Gpu_brand"].index("Intel"))

if st.button("Predict price"):
    x_res, y_res = (int(v) for v in resolution.split("x"))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

    # Same feature engineering as the notebook
    row = {
        "Ram": ram, "Weight": weight, "Touchscreen": int(touchscreen), "IPS": int(ips),
        "PPI": ppi, "Cpu_GHz": cpu_ghz, "SSD": ssd, "HDD": hdd,
    }
    # One-hot columns exist as e.g. "Company_Dell" -> set the chosen one to 1
    for prefix, value in [("Company", company), ("TypeName", type_name), ("Cpu_brand", cpu_brand),
                          ("Gpu_brand", gpu_brand), ("OS", os_name)]:
        row[f"{prefix}_{value}"] = 1

    # Put the row in exactly the column order used during training; missing columns become 0
    x_new = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    price = model.predict(x_new)[0]
    st.success(f"Estimated price: ₹ {price:,.0f}")
    st.caption("This is an estimate from a machine learning model trained on about 1,200 laptops.")