import numpy as np
import matplotlib.pyplot as plt


def img2mat(img):
    # flatten image
    d1, d2, d3 = img.shape
    out = img.reshape(d1*d2, d3)
    return out

def get_color_transfered_centers(T, input_centers, ref_centers, a):
    # calculate new color centroids base on the Transport Matrix T
    X = np.zeros(input_centers.shape)
    for i, _ in enumerate(input_centers):
        d = T[i, :].sum()
        if not d: d = a[i]
        X[i, :] = np.dot(T[i, :], ref_centers)/d
    return X

# color transform image
def update_image(labels, new_centers, input_mat):
    # transform image to new colors
    out = np.zeros(input_mat.shape)
    for i, l in enumerate(labels):
        out[i, :] = new_centers[l, :]
    return out.astype(int)

def plot_triple(input_img, reference_img, ct_img):
    fig, axes = plt.subplots(ncols=3, figsize=(16, 9))

    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[1].imshow(reference_img)
    axes[1].set_title('Reference')
    axes[2].imshow(ct_img)
    axes[2].set_title('Color Transferred Image')

    return fig
