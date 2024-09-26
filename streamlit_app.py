import streamlit as st
import importlib

# Configuration for the main app
st.set_page_config(page_title="Lab Manager")

# Define the pages, mapping them to their module names
pages = {
    "Lab 01": "lab-01",
    "Lab 02": "lab-02",
    "Lab 03": "lab-03",
    "Lab 04": "lab-04.py",
    "Lab 05": "lab-05",
    "Lab 06": "lab-06"
}

# Sidebar for navigation
st.sidebar.title("Lab Selector")

# Set default selection to "Lab 04"
default_page = "Lab 04"
selection = st.sidebar.radio("Select Lab", list(pages.keys()), index=list(pages.keys()).index(default_page))

# Dynamically import and run the selected lab page
try:
    page_module = importlib.import_module(pages[selection])
    page_module.main()
except ModuleNotFoundError:
    st.error(f"Module for {selection} not found.")
except AttributeError:
    st.error(f"Module {pages[selection]} does not have a main() function.")
