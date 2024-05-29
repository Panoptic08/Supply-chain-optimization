import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq
import numpy as np



# Load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# Load the entire dataset
data = load_data('generated_dataset.csv')

# Access individual columns
product_type = data['Product type']
sku = data['SKU']
price = data['Price']
availability = data['Availability']
number_of_products_sold = data['Number of products sold']
revenue_generated = data['Revenue generated']
customer_demographics = data['Customer demographics']
stock_levels = data['Stock levels']
lead_times = data['Lead times']
order_quantities = data['Order quantities']
shipping_times = data['Shipping times']
shipping_carriers = data['Shipping carriers']
shipping_costs = data['Shipping costs']
supplier_name = data['Supplier name']
location = data['Location']
lead_time = data['Lead time']
production_volumes = data['Production volumes']
manufacturing_lead_time = data['Manufacturing lead time']
manufacturing_costs = data['Manufacturing costs']
inspection_results = data['Inspection results']
defect_rates = data['Defect rates']
transportation_modes = data['Transportation modes']
routes = data['Routes']
costs = data['Costs']

# Streamlit app
st.title('Supply Chain Optimization Dashboard')

# Sidebar
st.sidebar.title('Select Visualization')

# Visualization selection
visualization_choice = st.sidebar.radio('Select Visualization', ['Supplier Performance', 
                                                                  'Supplier Share',
                                                                  'Product Information', 'Customer Insights', 
                                                                  'Inventory Management', 'Supplier Supervision', 
                                                                  'Shipping and Logistics', 'Data Analysis', 'ML Data Analysis'])

# Load the entire dataset
data = load_data('generated_dataset.csv')

# Visualization
if visualization_choice == 'Supplier Performance':
    st.subheader('Supplier Performance')
    selected_supplier = st.sidebar.selectbox('Select Supplier', data['Supplier name'].unique())
    # Filter data for selected supplier
    supplier_performance = data[data['Supplier name'] == selected_supplier]
    
    # Count total number of all products for the selected supplier
    total_available_products = supplier_performance['Availability'].sum()
    total_units_sold = supplier_performance['Number of products sold'].sum()
    
    # Count total available products for skincare, haircare, and cosmetics categories
    skincare_available = supplier_performance[supplier_performance['Product type'] == 'skincare']['Availability'].sum()
    haircare_available = supplier_performance[supplier_performance['Product type'] == 'haircare']['Availability'].sum()
    cosmetics_available = supplier_performance[supplier_performance['Product type'] == 'cosmetics']['Availability'].sum()
    
    skincare_sold = supplier_performance[supplier_performance['Product type'] == 'skincare']['Number of products sold'].sum()
    haircare_sold = supplier_performance[supplier_performance['Product type'] == 'haircare']['Number of products sold'].sum()
    cosmetics_sold = supplier_performance[supplier_performance['Product type'] == 'cosmetics']['Number of products sold'].sum()
    
    # Display the counts for skincare, haircare, and cosmetics
    
    # Display the counts
    st.write('Total Available Products for', selected_supplier, ':', total_available_products)
    st.write('Total Units Sold for', selected_supplier, ':', total_units_sold)
    
    
    st.write('Total Available Skincare Products for', selected_supplier, ':', skincare_available)
    st.write('Total Available Haircare Products for', selected_supplier, ':', haircare_available)
    st.write('Total Available Cosmetics Products for', selected_supplier, ':', cosmetics_available)
    
    st.write('Total Sold Skincare Products for', selected_supplier, ':', skincare_sold)
    st.write('Total Sold Haircare Products for', selected_supplier, ':', haircare_sold)
    st.write('Total Sold Cosmetics Products for', selected_supplier, ':', cosmetics_sold)
    
    
    
    # Create DataFrame for displaying data in a tabular format
    data_display = pd.DataFrame({
        'Category': ['Total Available Products', 'Total Units Sold', 'Total Available Skincare Products',
                     'Total Available Haircare Products', 'Total Available Cosmetics Products',
                     'Total Sold Skincare Products', 'Total Sold Haircare Products', 'Total Sold Cosmetics Products'],
        'Count': [total_available_products, total_units_sold, skincare_available,
                  haircare_available, cosmetics_available, skincare_sold,
                  haircare_sold, cosmetics_sold]
        
    })
    
    st.write(data_display)



elif visualization_choice == 'Supplier Share':
    # Revenue share
    st.subheader('Category Wise Share of Top 5 Suppliers')
    top5_revenue = data.groupby('Supplier name')['Revenue generated'].sum().nlargest(5)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].pie(top5_revenue, labels=top5_revenue.index, autopct='%1.1f%%')
    ax[0].set_title('Revenue Share')

    # Inventory share
    top5_inventory = data.groupby('Supplier name')['Stock levels'].sum().nlargest(5)
    ax[1].pie(top5_inventory, labels=top5_inventory.index, autopct='%1.1f%%')
    ax[1].set_title('Inventory Share')

    # Costs share
    top5_costs = data.groupby('Supplier name')['Costs'].sum().nlargest(5)
    ax[2].pie(top5_costs, labels=top5_costs.index, autopct='%1.1f%%')
    ax[2].set_title('Costs Share')

    # Adjust layout
    plt.tight_layout()

    st.pyplot(fig)


# Display subcategories and data based on the selected main category
elif visualization_choice == 'Product Information':
    st.subheader('Product Information')
    
    
    # Count the number of products in each category
    num_skincare_products_a = data[data['Product type'] == 'skincare']['Availability'].sum()
    num_haircare_products_a = data[data['Product type'] == 'haircare']['Availability'].sum()
    num_cosmetics_products_a = data[data['Product type'] == 'cosmetics']['Availability'].sum()
    
    # Calculate total number of available products and units sold
    total_available_products = data['Availability'].sum()
    total_units_sold = data['Number of products sold'].sum()
    
    num_skincare_products_s = data[data['Product type'] == 'skincare']['Number of products sold'].sum()
    num_haircare_products_s = data[data['Product type'] == 'haircare']['Number of products sold'].sum()
    num_cosmetics_products_s = data[data['Product type'] == 'cosmetics']['Number of products sold'].sum()
    
    # Count the number of products in each category
    num_skincare_products = num_skincare_products_a + num_skincare_products_s
    num_haircare_products = num_haircare_products_a + num_haircare_products_s
    num_cosmetics_products = num_cosmetics_products_a + num_cosmetics_products_s
    
    
        # Display the counts in a tabular format
    with st.expander('View Counts'):
        data_display = pd.DataFrame({
            'Category': ['Skincare', 'Haircare', 'Cosmetics', 'Total Available', 'Total Sold'],
            'Total Products': [num_skincare_products, num_haircare_products, num_cosmetics_products, total_available_products, total_units_sold],
            'Available Products': [num_skincare_products_a, num_haircare_products_a, num_cosmetics_products_a, '-', '-'],
            'Sold Products': [num_skincare_products_s, num_haircare_products_s, num_cosmetics_products_s, '-', '-']
        })
        st.write(data_display)
    
    # Display the counts
    st.write('Number of Skincare Products:', num_skincare_products)
    st.write('Number of Haircare Products:', num_haircare_products)
    st.write('Number of Cosmetics Products:', num_cosmetics_products)
    st.write('Total Available Products:', total_available_products)
    st.write('Total Units Sold:', total_units_sold)
    st.write('Number of Skincare Products Available:', num_skincare_products_a)
    st.write('Number of Haircare Products Available:', num_haircare_products_a)
    st.write('Number of Cosmetics Products Available:', num_cosmetics_products_a)
    st.write('Number of Skincare Products Sold:', num_skincare_products_s)
    st.write('Number of Haircare Products Sold:', num_haircare_products_s)
    st.write('Number of Cosmetics Products Sold:', num_cosmetics_products_s)
    


elif visualization_choice == 'Customer Insights':
    st.subheader('Customer Insights')
    num_female = (data['Customer demographics'] == 'Female').sum()
    num_male = (data['Customer demographics'] == 'Male').sum()
    num_un = (data['Customer demographics'] == 'Unknown').sum()
    st.write('Number of Females:- ', num_female)
    st.write('Number of Males:- ', num_male)
    st.write('Number of Unknowns:- ', num_un)

    # Create a pie chart for customer demographics
    labels = ['Female', 'Male', 'Unknown']
    sizes = [num_female, num_male, num_un]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)  # explode 1st slice (Female)

    # Set the size of the figure
    plt.figure(figsize=(6, 6))

    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)



elif visualization_choice == 'Inventory Management':
    st.subheader('Inventory Management')

    # Group data by supplier and product type, then sum the stock levels
    inventory_data = data.groupby(['Supplier name', 'Product type'])['Stock levels'].sum().unstack()

    # Fill NaN values with 0
    inventory_data = inventory_data.fillna(0)

    # Display the inventory data as a DataFrame
    st.write(inventory_data)

    # Create a single bar graph for all suppliers with separate colors and spaces between categories
    plt.figure(figsize=(10, 6))

    # Get the number of suppliers and product categories
    num_suppliers = len(inventory_data)
    num_categories = len(inventory_data.columns)

    # Define colors for each supplier
    colors = plt.cm.tab10(np.linspace(0, 1, num_suppliers))

    # Create positions for each bar with spaces between categories
    positions = np.arange(num_categories) * (num_suppliers + 1)

    # Plot each supplier's data with a different color
    for i, supplier in enumerate(inventory_data.index):
        supplier_inventory = inventory_data.loc[supplier]
        plt.bar(positions + i, supplier_inventory, width=1, label=supplier, color=colors[i])

    # Customize the plot
    plt.xlabel('Product Category')
    plt.ylabel('Stock Levels')
    plt.title('Inventory Levels by Supplier and Product Category')
    plt.xticks(positions + (num_suppliers - 1) / 2, inventory_data.columns)
    plt.legend()

    # Display the plot
    st.pyplot(plt)

elif visualization_choice == 'Supplier Supervision':
    st.subheader('Supplier Supervision')
    st.write(data[['Supplier name', 'Location', 'Lead time', 'Inspection results', 
                   'Defect rates', 'Costs']])

elif visualization_choice == 'Shipping and Logistics':
    st.subheader('Shipping and Logistics')
    st.write(data[['Shipping times', 'Shipping carriers', 'Shipping costs', 
                   'Transportation modes', 'Routes', 'Costs']])
    
# Data Analysis section
elif visualization_choice == 'Data Analysis':
    st.title('Supply Chain Data Analysis')

    # Display dataset information
    st.subheader("Dataset Information")
    st.write(data.info())

    # Display a sample of the dataset
    st.subheader("Sample of the Dataset")
    st.write(data.head())

    # Display descriptive statistics
    st.subheader("Descriptive Statistics")
    st.write(data.describe())

    # Display correlation matrix
    st.subheader("Correlation Matrix")
    numeric_data = data.select_dtypes(include=['number'])  # Select only numeric columns
    st.write(numeric_data.corr())

# ML Data Analysis section
elif visualization_choice == 'ML Data Analysis':
    st.title('Supply Chain Optimization with Machine Learning')

    # Perform clustering analysis
    features = ['Price', 'Availability', 'Number of products sold', 'Revenue generated', 
                'Stock levels', 'Lead times', 'Order quantities', 'Shipping times', 'Shipping costs',
                'Lead time', 'Production volumes', 'Manufacturing lead time', 'Manufacturing costs', 'Defect rates', 'Costs']

    X = data[features]

    # Perform KMeans clustering
    num_clusters = 3  # Update to desired number of clusters
    centroids, _ = kmeans(X, num_clusters)
    cluster_ids, _ = vq(X, centroids)
    data['Cluster'] = cluster_ids

    # List of unique suppliers
    suppliers = data['Supplier name'].unique()
    products = data['Product type'].unique()

        # For each supplier
    for supplier in suppliers:
        st.subheader(f"Analysis for Supplier: {supplier}")
        
        # Filter data for the current supplier
        supplier_data = data[data['Supplier name'] == supplier]
        
        # Perform clustering analysis for the current supplier
        X_supplier = supplier_data[features]
        centroids_supplier, _ = kmeans(X_supplier, num_clusters)  # Use the specified number of clusters
        cluster_ids_supplier, _ = vq(X_supplier, centroids_supplier)
        supplier_data['Cluster'] = cluster_ids_supplier
        
        # Display cluster analysis for the current supplier
        st.subheader("Cluster Analysis")
        st.write(supplier_data.groupby('Cluster')[features].mean())
        
        # Calculate the mean revenue generated and lead time for each cluster and determine the optimal cluster for revenue
        cluster_means_supplier = supplier_data.groupby('Cluster')[['Revenue generated', 'Lead time']].mean()
        optimal_cluster_revenue_supplier = cluster_means_supplier['Revenue generated'].idxmax()

        # Calculate the mean revenue generated and lead time for each cluster and determine the optimal cluster for lead time
        optimal_cluster_lead_time_supplier = cluster_means_supplier['Lead time'].idxmax()
        
        # Display optimal cluster for revenue and lead time separately
        st.subheader("Optimal Cluster for Revenue Generated")
        st.write(f"Cluster {optimal_cluster_revenue_supplier}")

        st.subheader("Optimal Cluster for Lead Time")
        st.write(f"Cluster {optimal_cluster_lead_time_supplier}")


    # For each product type
    for product_type in products:
        st.subheader(f"Analysis for {product_type.capitalize()} Products")
        
        # Filter data for the current product type
        product_data = data[data['Product type'] == product_type]

        # Perform clustering analysis for the current product type
        X_product = product_data[features]
        centroids_product, _ = kmeans(X_product, num_clusters)  # Use the specified number of clusters
        cluster_ids_product, _ = vq(X_product, centroids_product)
        product_data['Cluster'] = cluster_ids_product

        # Display cluster analysis for the current product type
        st.subheader(f"Cluster Analysis for {product_type.capitalize()} Products")
        st.write(product_data.groupby('Cluster')[features].mean())
        
        # Calculate the mean revenue generated and lead time for each cluster and determine the optimal cluster for revenue
        cluster_means_product = product_data.groupby('Cluster')[['Revenue generated', 'Lead time']].mean()
        optimal_cluster_revenue_product = cluster_means_product['Revenue generated'].idxmax()

        # Calculate the mean revenue generated and lead time for each cluster and determine the optimal cluster for lead time
        optimal_cluster_lead_time_product = cluster_means_product['Lead time'].idxmax()
        
        # Display optimal cluster for revenue and lead time separately
        st.subheader("Optimal Cluster for Revenue Generated")
        st.write(f"Cluster {optimal_cluster_revenue_product}")

        st.subheader("Optimal Cluster for Lead Time")
        st.write(f"Cluster {optimal_cluster_lead_time_product}")