import streamlit as st 

from social_monitoring import display 
#from survey_monitoring import display as survey_display

# Set the title of the app
st.set_page_config(page_title="Data Quality Monitoring Tool", layout="wide")
#st.title("Data Monitoring Dashboard")
st.markdown("""
    <h1 style='text-align: center; color: blue; font-weight: bold; font-family: "Courier New", monospace;'>
    RedditIQ:</h1> <h4 style= color: black;> Social Data Quality Monitoring Tool
    </h4>
    """, unsafe_allow_html=True)
st.write("")
# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = None

# Display the buttons only if no page is selected
if st.session_state.page is None:
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Social Data Monitoring"):
            st.session_state.page = "social"
    with col2:
            st.image("dd.jpg", caption="", width=650)

# Render the selected page based on the button clicked
if st.session_state.page == "social":
#   st.subheader("Social Data Monitoring Page")
   display()  # Call your function for social monitoring
    # Add a 'Go Back' button to navigate back to the main dashboard
   if st.button("Go Back"):
      st.session_state.page = None

#if st.session_state.page == "survey":
 #  st.subheader("Survey Data Monitoring Page")
 #  survey_display()  # Call your function for survey monitoring
    # Add a 'Go Back' button to navigate back to the main dashboard
 #  if st.button("Go Back"):
  #    st.session_state.page = None

# Display message if no page is selected
if st.session_state.page is None:
    st.write("Please select button to monitor Reddit data.")

