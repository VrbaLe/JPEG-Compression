import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

# Standardna luminancna kvantizaciona matrica
Q_std = np.array([[16,11,10,16,24,40,51,61],
                  [12,12,14,19,26,58,60,55],
                  [14,13,16,24,40,57,69,56],
                  [14,17,22,29,51,87,80,62],
                  [18,22,37,56,68,109,103,77],
                  [24,35,55,64,81,104,113,92],
                  [49,64,78,87,103,121,120,101],
                  [72,92,95,98,112,100,103,99]])

def scale_quant(Q, quality):
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2*quality
    Q_new = np.floor((Q*scale + 50)/100)
    Q_new[Q_new==0] = 1
    return Q_new

def block_process(img, func):
    h, w = img.shape
    # Povećaj dimenzije slike tako da budu deljive sa 8 radi obrade po 8x8 blokovima
    H, W = (h + 7)//8*8, (w + 7)//8*8
    padded = np.zeros((H,W))
    padded[:h,:w] = img
    out = np.zeros((H,W))
    for i in range(0, H, 8):
        for j in range(0, W, 8):
            out[i:i+8,j:j+8] = func(padded[i:i+8,j:j+8])
    return out[:h,:w]


def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# --- Parametri ---
quality = 30  # 1–100, manji broj = vise kompresije
Q = scale_quant(Q_std, quality)

# Učitaj sliku i konvertuj u YCbCr
img = Image.open("input2.jpg").convert("YCbCr")
y, cb, cr = [np.array(c, dtype=float)-128 for c in img.split()]

# DCT + kvantizacija
y_dct = block_process(y, lambda b: np.round(dct2(b)/Q))
cb_dct = block_process(cb, lambda b: np.round(dct2(b)/Q))
cr_dct = block_process(cr, lambda b: np.round(dct2(b)/Q))

# IDCT + dekvantizacija
y_rec = block_process(y_dct, lambda b: idct2(b*Q)) + 128
cb_rec = block_process(cb_dct, lambda b: idct2(b*Q)) + 128
cr_rec = block_process(cr_dct, lambda b: idct2(b*Q)) + 128

# Rekombinuj i sačuvaj
img_rec = Image.merge("YCbCr", [Image.fromarray(np.clip(c,0,255).astype(np.uint8)) for c in [y_rec, cb_rec, cr_rec]])
img_rec.convert("RGB").save("output.jpg", "JPEG")
