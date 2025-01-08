import pandas as pd
import numpy as np
from PIL import Image

# Load the CSV file
data = pd.read_csv('./data/sign_mnist_train.csv')

# Select the first row (for example)
first_image_data = data.iloc[0, 1:].values  # Skip the label (column 0)

# Reshape the data into a 28x28 matrix and scale it properly
first_image_data = first_image_data.reshape(28, 28).astype(np.uint8)

# Create a PIL image from the numpy array
img = Image.fromarray(first_image_data)

# Resize the image to double-check how it looks at higher resolution
# img = img.resize((280, 280), Image.NEAREST)

# Save the image as a PNG
img.save('./testImage/test_image.png')

# Optionally show the image using PIL to visually inspect it
img.show()
