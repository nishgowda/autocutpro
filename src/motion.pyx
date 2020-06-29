import cython
cdef extern from "<complex>":
    double abs(double complex)
cpdef double compare_rgb(unsigned char [:, :, :] img, unsigned char [:, :, :] prevImg):
    # set the variable extension types
    cdef int x, y, w, h
    cdef unsigned char colorsB1, colorsG1, colorsR1, colorsB2, colorsG2, colorsR2
    cdef double diffR = 0.0, diffG = 0.0, diffB = 0.0

    # grab the image dimensions
    h = img.shape[0]
    w = img.shape[1]
    numPixels = img.size
    # loop over the image
    for x in range(w):
        for y in range(h):
          colorsB1 = img[x, y, 0]
          colorsG1 = img[x, y, 1]
          colorsR1 = img[x, y, 2]
          colorsB2 = prevImg[x, y, 0]
          colorsG2 = prevImg[x, y, 1]
          colorsR2 = prevImg[x, y, 2]
          diffR += abs(colorsR1 - colorsR2) / 255.0
          diffG += abs(colorsG1 - colorsG2) / 255.0
          diffB += abs(colorsB1 - colorsB2) / 255.0
    diffR /= numPixels
    diffG /= numPixels
    diffB /= numPixels
    return (diffR + diffG + diffB) / 3.0
