import streamlit as st
import pandas as pd

st.set_page_config(
    page_title = "Data Page"
)

st.title("Data")

excel_files = ["bakery_income_sales_data_2022.xlsx", 
               "bakery_income_sales_data_2023.xlsx", 
               "bakery_income_sales_data_2024.xlsx",
               "bakery_operations_data_2022.xlsx", 
               "bakery_operations_data_2023.xlsx",
               "bakery_operations_data_2024.xlsx",
               "bakery_suppliers_data_2022.xlsx",
              "bakery_suppliers_data_2023.xlsx",
              "bakery_suppliers_data_2024.xlsx",
               "bakery_wages_data_2022.xlsx",
              "bakery_wages_data_2023.xlsx",
              "bakery_wages_data_2024.xlsx"]

for file in excel_files:
    data = pd.read_excel("Bakery Data/"+file)
    file_name = ((' '.join((file[7:]).split('_'))).title()).split(".")[0]
    st.write(file_name)
    st.write(data)