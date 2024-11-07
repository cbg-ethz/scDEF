import matplotlib.colors as mc
import colorsys


def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = list(colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2]))
    for i in range(3):
        c[i] = max(0.0, min(1.0, c[i]))
    return c
