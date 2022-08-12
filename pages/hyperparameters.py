from asyncio import SafeChildWatcher
import streamlit as st


st.write("## Hyperparameters")
st.write("""##### Color centroids number
Number of color centroid used to transfer the color information between the images, it's correspond to the parameter k in k-means algorithm used to calculate the centroids""")
st.write("""##### Number of Epochs
Number of complete iteration that algorithm take to complete the color transfer""")
st.write("""##### Type of algorithm for the colo transfer
There are three type of algorithm implemented for the transfer color problem, all three are different vesion of the original Frank-Wolfe algorithm:  
1. Classic Frank-Wolfe algorithm
2. Block coordinate Frank-Wolfe algorithm
3. Block coordinate Away step Frank-Wolfe algorithm""")