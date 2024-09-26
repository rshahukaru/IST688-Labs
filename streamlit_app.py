import streamlit as st

# Set up individual pages for each homework
lab_01_page = st.Page("lab_01.py", title="Lab-01")
lab_02_page = st.Page("lab_02.py", title="Lab-02")
lab_03_page = st.Page("lab_03.py", title="Lab-03")
lab_04_page = st.Page("lab_04.py", title="Lab-04", default = True)

# The below labs' .py files have not been created yet. So, I am redirecting them to lab-04.py
lab_05_page = st.Page("lab_05.py", title="Lab-05")
lab_06_page = st.Page("lab_06.py", title="Lab-06")

# Navigation setup with all homework pages
pg = st.navigation([
    lab_01_page, lab_02_page, lab_03_page, lab_04_page, lab_05_page,
    lab_06_page
])

# Configuration of the main app
st.set_page_config(page_title="Lab Assignments")

# Running the page navigation
pg.run()
