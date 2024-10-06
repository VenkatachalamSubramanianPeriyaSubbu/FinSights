import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)
def plot_data(path, plot_type):
    df = pd.read_excel(path)

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'Bar':
        category_summary = df.groupby('Category')['Amount'].sum().reset_index()
        ax.bar(category_summary['Category'], category_summary['Amount'], color='skyblue')
        ax.set_ylabel('Total Amount', fontweight='bold')
        ax.set_xlabel('Category', fontweight='bold')
        ax.set_title('Total Amount by Category', fontweight='bold', size = 15)
    elif plot_type == 'Line':
        category_summary = df.groupby('Date')['Amount'].sum().reset_index()
        ax.plot(category_summary['Date'], category_summary['Amount'], marker='o', color='b')
        ax.set_ylabel('Total Amount', fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_title('Total Amount earned over time', fontweight='bold', size = 15)
    else:
        st.error("plot_type must be 'bar' or 'line'")
        return
    plt.xticks(rotation=45, ha='right')

df = pd.DataFrame({"Year":['2022'] * 4 + ['2023'] * 4 + ['2024'] * 4, 
                  "Category": ['Income/Sales', 'Suppliers', 'Operating Costs', 'Wages']*3,
                  "Path": ['bakery_income_sales_data_2022.xlsx', 'bakery_suppliers_data_2022.xlsx', 'bakery_operations_data_2022.xlsx', 'bakery_wages_data_2022.xlsx',
                           'bakery_income_sales_data_2023.xlsx', 'bakery_suppliers_data_2023.xlsx', 'bakery_operations_data_2023.xlsx', 'bakery_wages_data_2023.xlsx',
                           'bakery_income_sales_data_2024.xlsx', 'bakery_suppliers_data_2024.xlsx', 'bakery_operations_data_2024.xlsx', 'bakery_wages_data_2024.xlsx']})

st.set_page_config(
    page_title = "Visualization Page"
)

st.title("Dashboard")

selected_category = st.radio("Select a Category:", ['Income/Sales', 'Suppliers', 'Operating Costs', 'Wages'])
selected_chart = st.radio("Select a Chart:", ['Bar', 'Line'])
selected_year = st.radio("Year:", ['2022','2023','2024'])
st.subheader(f'{selected_category} for the year {selected_year}')
filepath = 'Bakery Data/'+df[(df['Year']==selected_year) & (df['Category']==selected_category)]['Path'].values[0]
st.pyplot(plot_data(filepath, selected_chart))