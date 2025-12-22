import numpy as np
import matplotlib.pyplot as plt

# Load 768 floats
frame = np.loadtxt("D:/Fuego/python/textFiles/tester.txt")

assert frame.size == 768, "Expected 768 values"

# MLX90640 is 24 rows x 32 columns
image = frame.reshape((24, 32))

plt.figure(figsize=(6, 4))
plt.imshow(image, cmap="inferno")
plt.colorbar(label="Temperature (°C)")
plt.title("MLX90640 Thermal Image")
plt.axis("off")
plt.show()