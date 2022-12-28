import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# load the image into a NumPy array
image = np.array(Image.open('image.jpg'))

# compute the fast Fourier transform of the image
fourier = np.fft.fft2(image)

# shift the zero frequency component to the center of the spectrum
fourier_shift = np.fft.fftshift(fourier)

# take the magnitude of the transformed image
magnitude = np.abs(fourier_shift)

# plot the magnitude of the transformed image
#plt.imshow(magnitude, cmap='gray')
plt.imshow((magnitude * 255).astype(np.uint8))
plt.show()
