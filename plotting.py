import matplotlib.pyplot as plt

def plot_data(path, plot_type):
    df = pd.read_excel(path)

    fig, ax = plt.subplots(figsize=(10, 6))

    if plot_type == 'bar':
        category_summary = df.groupby('Category')['Amount'].sum().reset_index()
        ax.bar(category_summary['Category'], category_summary['Amount'], color='skyblue')
        ax.set_ylabel('Total Amount', fontweight='bold')
        ax.set_xlabel('Category', fontweight='bold')
        ax.set_title('Total Amount by Category', fontweight='bold', size = 15)
    elif plot_type == 'line':
        category_summary = df.groupby('Date')['Amount'].sum().reset_index()
        ax.plot(category_summary['Date'], category_summary['Amount'], marker='o', color='b')
        ax.set_ylabel('Total Amount', fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_title('Total Amount earned over time', fontweight='bold', size = 15)
    else:
        st.error("plot_type must be 'bar' or 'line'")
        return
    plt.xticks(rotation=45, ha='right')