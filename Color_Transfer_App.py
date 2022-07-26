import streamlit as st

import fw
import bcfw
import asfw
from utility import from_png_to_jpeg

import image_processing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from PIL import Image
from matplotlib import image as mpimg

from io import BytesIO # to resize the plots in Streamlit
import os # to convert png images in jpeg format


with st.sidebar:
    st.write("#### Hypeparameters:")
    n_clusters = st.slider("Select the number of color centroids", 1, 32, value=16)
    num_epochs = st.number_input('Set the number of epochs', 10, int(1e7), value=10000)
    fw_type = st.selectbox('Choose the optimization Frank-Wolfe algorithm',
     ('Classic Frank-Wolfe', 'Block-Coordinate Frank-Wolfe', 'Block-Coordinate Away-Step Frank-Wolfe'))
    print_plots = st.checkbox("Show the error plot",)

st.title("Color Trasfer app")
st.write(
    """This app shows how you can use the Frank Wolfe Optimization algorithm to perform a color transfer between a reference image to an input image, like the example below."""
)

st.image("img/example.jpg")


st.write(">More details and the link of the realted paper in the About page on the left menu")

st.write(
    """ ## Transfer your image
    You can play with the hyperparameters in the left column and upload your reference and input images to make your personal color transfer"""
)



col1, col2 = st.columns(2)

with col1:
    st.write('#### Upload your own input file:')
    input_file = st.file_uploader(label='Upload an image you would like color transform:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    fig, ax = plt.subplots(1)
    if input_file is None:
        input_image = mpimg.imread("img/blue.jpg")
    else:
        input_image = mpimg.imread(input_file)
        if isinstance(input_image[0,0,0],np.float32):  # if is png convert to jpeg (from [0,1] to [0,256])               
            input_image = from_png_to_jpeg(input_image)

    ax.imshow(input_image)
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.write('#### Upload your own reference file:')
    reference_file = st.file_uploader(label='Upload an image you would like as reference:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)  


    fig, ax = plt.subplots(1)
    if reference_file is None:
        ref_image = mpimg.imread("img/reading.jpg")
    else:
        ref_image = mpimg.imread(reference_file)
        if isinstance(ref_image[0,0,0],np.float32): 
            ref_image = from_png_to_jpeg(ref_image)

    ax.imshow(ref_image)
    ax.axis('off')
    st.pyplot(fig)

st.write("According to the chosen settings, the procedure can take various minute to complete. (range to find)") 

start = st.button("Start Color Transfer")

ct = st.container()

st.caption("Created by: [Federico Bottarelli](https://github.com/federicobottarelli), [Jan Elfes](https://github.com/jelfes) and [Ivan Padezhki](https://github.com/ivanpadezhki)")


with ct:
    if(start):
        st.text("Start clustering of the color centroids ...")

        ## get color centroids via kmeans
        ref_mat = image_processing.img2mat(ref_image)
        input_mat = image_processing.img2mat(input_image)


        # fit kmeans
        kmeans32_ref = KMeans(n_clusters=n_clusters)
        kmeans32_ref.fit(ref_mat)

        st.text("... half done with the clustering ...")

        kmeans32_in = KMeans(n_clusters=n_clusters)
        kmeans32_in.fit(input_mat)

        # get coordinates
        Y = kmeans32_ref.cluster_centers_
        X = kmeans32_in.cluster_centers_

        # create our input and output distributions (a and b respectively).
        _, c = np.unique(kmeans32_ref.labels_, return_counts=True)
        b = c/c.sum()

        _, c = np.unique(kmeans32_in.labels_, return_counts=True)
        a = c/c.sum()

        cl_input = kmeans32_in.labels_

        # create costmatrix C
        C = np.zeros((a.size, b.size))
        for i, x_i in enumerate(X):
            for j, y_j in enumerate(Y):
                C[i, j] = np.linalg.norm(x_i-y_j)

        st.text("Start color transfer ...")
        # perform Color Transport on the input image with FW.

        # set first dimension of T to b and all other entries to 0, then T is guaranteed to be feasible
        T_init = np.zeros((a.size, b.size))
        T_init[0, :] = b

        # Do Color Transport ('Classic Frank-Wolfe', 'Block-Coordinate Frank-Wolfe', 'Block-Coordinate Away-Step Frank-Wolfe')
        if fw_type == "Classic Frank-Wolfe":
            T_fw, iteration_counter, err, gradient, t = fw.min_fw(var=T_init, a=a, b=b, C=C, epoch=num_epochs)
            
        elif fw_type == "Block-Coordinate Frank-Wolfe":
            T_fw, iteration_counter, err, gradient, t = bcfw.min_block_fw(var=T_init, a=a, b=b, C=C, epoch=num_epochs)

        else:
            T_fw, iteration_counter, err, aw_number, _, gradient, drop_steps, _, t = asfw.min_block_away_fw_ELS(var=T_init, a=a, b=b, C=C, epoch=num_epochs)

        # Get color transferred image
        new_centers = image_processing.get_color_transfered_centers(T_fw, X, Y, a)
        ct_mat = image_processing.update_image(cl_input, new_centers, input_mat)
        ct_image = ct_mat.reshape(input_image.shape)

        st.text("Done!")
        if (fw_type == "Classic Frank-Wolfe") or (fw_type == "Block-Coordinate Frank-Wolfe"):
            st.write(f'Number of Iterations: {iteration_counter} \tError: {np.round(err[-1], 2)}, \t time: {np.round(t, 2)}s')
        else:
            st.write(f'Number of Iterations: {iteration_counter} \tError: {np.round(err[-1], 2)}, \t time: {np.round(t, 2)}s \taway-steps: {aw_number} \tdrop-steps: {drop_steps}')

        fig, ax = plt.subplots(1)
        ax.imshow(ct_image)
        ax.axis('off')
        st.pyplot(fig)

                # print error plots if requested
        if print_plots:
            st.write("#### Error plot")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(err)
            ax.set_yscale('log')
            buf = BytesIO() # those lines are necessary for resizing the plot
            fig.savefig(buf, format="png")
            st.image(buf)
            # st.pyplot(fig)
