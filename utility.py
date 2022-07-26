
def from_png_to_jpeg(input_img):
    image_new = (input_img*256).astype(int)
    return image_new