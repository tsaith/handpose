from PIL import Image


def make_square(im, mode='RGB', bg_color='black'):
    # Make a square image.

    w, h = im.size
    w_out = max(w, h)
    h_out = w_out
    im_out = Image.new(mode, (w_out, w_out), bg_color)
    im_out.paste(im, (int((w_out - w) / 2), int((h_out - h) / 2)))

    return im_out


