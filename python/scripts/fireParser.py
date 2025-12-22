import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

def read_thermal_data(file_path):
    """
    Read 768 float values from a binary file containing MLX90640 thermal data.
    """
    with open(file_path, 'rb') as f:
        # Read 768 floats (each float is 4 bytes)
        data = f.read(768 * 4)
        if len(data) != 768 * 4:
            raise ValueError(f"File does not contain exactly 768 floats: {len(data)//4} found")

        # Unpack as little-endian floats
        temperatures = struct.unpack('<768f', data)
        return np.array(temperatures)

def visualize_thermal_image(data, output_path=None):
    """
    Visualize the thermal data as a 32x24 image.
    """
    # Reshape to 32x24 (32 rows, 24 columns)
    image = data.reshape(32, 24)

    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title('MLX90640 Thermal Image')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    if output_path:
        plt.savefig(output_path)
        print(f"Image saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fireParser.py <binary_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        data = read_thermal_data(file_path)
        visualize_thermal_image(data)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)