import streamlit as st
import fw
import image_processing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from PIL import Image
from matplotlib import image as mpimg



with st.sidebar:
    n_clusters = st.slider("Select the number of color centroids", 1, 32, value=16)
    num_epochs = st.number_input('Set the number of epochs', 10, int(1e7), value=10000)

col1, col2 = st.columns(2)

with col1:
    st.header('Upload your own input file:')
    input_file = st.file_uploader(label='Upload an image you would like color transform:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    fig, ax = plt.subplots(1)
    if input_file is None:
        input_image = mpimg.imread("blue.jpg")
    else:
        input_image = mpimg.imread(input_file)

    ax.imshow(input_image)
    ax.axis('off')
    st.pyplot(fig)

with col2:
    st.header('Upload your own reference file:')
    reference_file = st.file_uploader(label='Upload an image you would like as reference:', type=['jpg', 'jpeg', 'png'], accept_multiple_files=False)

    fig, ax = plt.subplots(1)
    if reference_file is None:
        ref_image = mpimg.imread("reading.jpg")
    else:
        ref_image = mpimg.imread(reference_file)

    ax.imshow(ref_image)
    ax.axis('off')
    st.pyplot(fig)

start = st.button("Start Color Transfer")

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

    # Do Color Transport
    T_fw, iteration_counter, err, gradient, t = fw.min_fw(var=T_init, a=a, b=b, C=C, epoch=num_epochs)

    # Get color transferred image
    new_centers = image_processing.get_color_transfered_centers(T_fw, X, Y, a)
    ct_mat = image_processing.update_image(cl_input, new_centers, input_mat)
    ct_image = ct_mat.reshape(input_image.shape)

    st.text("Done!")
    st.write(f'Number of Iterations: {iteration_counter} \tError: {np.round(err[-1], 2)}, \t time: {np.round(t, 2)}s')
    fig, ax = plt.subplots(1)
    ax.imshow(ct_image)
    ax.axis('off')
    st.pyplot(fig)

    
