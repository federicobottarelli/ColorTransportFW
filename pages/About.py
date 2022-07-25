import streamlit as st

st.write("## Color trasport problem")
st.write("this web application was made to show the results achieved for a data science optimization course project. ")
st.write("The paper with the detailed explanation of the algorithms used")

#st.download_button("Download PDF", "paper/FW_optimal_transport.pdf")
with open("paper/FW_optimal_transport.pdf", "rb") as file:
     btn = st.download_button(
             label="Download PDF",
             data=file,
             file_name="FW_optimal_transport.pdf"
           )