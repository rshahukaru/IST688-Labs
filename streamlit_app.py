import streamlit as st

# Set up individual pages for each homework
lab_01_page = st.Page("lab-01.py", title="Lab-01")
lab_02_page = st.Page("lab-02.py", title="Lab-02")
lab_03_page = st.Page("lab-03.py", title="Lab-03")
lab_04_page = st.Page("lab-04.py", title="Lab-04", default = True)

# The below labs' .py files have not been created yet. So, I am redirecting them to lab-04.py
lab_05_page = st.Page("lab-05.py", title="Lab-05")
lab_06_page = st.Page("lab-06.py", title="Lab-06")
lab_07_page = st.Page("lab-07.py", title="Lab-07")
lab_08_page = st.Page("lab-08.py", title="Lab-08")
lab_09_page = st.Page("lab-09.py", title="Lab-09")
lab_10_page = st.Page("lab-10.py", title="Lab-10")
lab_11_page = st.Page("lab-11.py", title="Lab-11")

# Navigation setup with all homework pages
pg = st.navigation([
    lab_01_page, lab_02_page, lab_03_page, lab_04_page, lab_05_page,
    lab_06_page, lab_07_page, lab_08_page, lab_09_page, lab_10_page,
    lab_11_page
])

# Configuration of the main app
st.set_page_config(page_title="Lab Assignments")

# Running the page navigation
pg.run()
