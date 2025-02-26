import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import base64

from collections import Counter
from typing import Tuple


class ImageUtils:
    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color code to an RGB tuple."""
        hex_color = hex_color.lstrip('#')  # Remove the '#' if it exists
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def get_greyscale_value(hex_color: str) -> float:
        """
        Convert a hex color to its greyscale value using the luminance formula:
        Luminance = 0.2989 * R + 0.5870 * G + 0.1140 * B
        """
        # Convert hex to RGB
        rgb = ImageUtils.hex_to_rgb(hex_color)
        
        # Apply the luminance formula
        r, g, b = rgb
        return 0.2989 * r + 0.5870 * g + 0.1140 * b

    @staticmethod    
    def img_to_base64(img):
        """Convert image to base64 string."""
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64


def render_latex_to_image(latex_code, latex_color):
    """Render the LaTeX code to an image with a given color."""
    
    # Calculate the greyscale value of the latex_color
    grey_value = ImageUtils.get_greyscale_value(latex_color)
    
    # Set the background color to black if the latex_color is closer to white
    if grey_value > 0.5:  # If the color is closer to white
        background_color = 'black'
    else:
        background_color = 'white'  # Keep the default white background if it's close to black
    
    # Create a figure with a dummy axis to calculate the height of the LaTeX text
    fig = plt.figure(figsize=(6, 6), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis('off')  # Hide the axes
    
    # Render LaTeX code with a temporary fontsize (we'll adjust the size later)
    text_obj = ax.text(0.5, 0.5, f'${latex_code}$', fontsize=50, ha='center', va='center', color=latex_color)

    # Get the bounding box of the text to estimate the height
    bbox = text_obj.get_window_extent()
    height_in_inches = bbox.height / fig.dpi  # Convert pixel height to inches based on dpi
    
    # Adjust figure size dynamically based on the text height
    fig.set_size_inches(6, height_in_inches * 1.2)  # Slightly increase the height for padding

    # Set background color
    plt.gca().set_facecolor(background_color)

    # Save to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    
    # Convert buffer to Image
    img = Image.open(buf)
    img = img.convert("RGBA")
    
    return img

def make_color_pixels_transparent(input_image, percentage_of_color_to_consider_majority=0.5, RGB=None) -> Tuple[io.BytesIO, Tuple[int, int, int, int]]:
    """
    Make pixels of the most common color in the image transparent.

    Args:
    - input_image: Either a PIL Image object or a BytesIO buffer containing the image.
    - percentage_of_color_to_consider_majority: The percentage of the most common color to trigger transparency.
    - RGB: A list or tuple with RGB values to use if no common color is found.

    Returns:
    - A tuple containing:
        - A BytesIO buffer with the modified image data.
        - The RGBA tuple of the most common color that was made transparent.
    """
    if percentage_of_color_to_consider_majority >= 1:
        raise ValueError("Percentage of color to consider majority must be less than 1")

    if isinstance(input_image, io.BytesIO):
        img = Image.open(input_image).convert("RGBA")  # Convert to RGBA to support transparency
    elif isinstance(input_image, Image.Image):
        img = input_image.convert("RGBA")  # Already a PIL image object
    else:
        raise ValueError("Input must be a PIL Image or a BytesIO buffer.")

    
    # Get the image data
    data = img.getdata()
    
    # Count the frequency of each color
    color_counts = Counter(data)
    
    # Get the most common color that appears less than 50% of the time
    total_pixels = len(data)
    most_common_color = None
    for color, count in color_counts.items():
        if count / total_pixels > percentage_of_color_to_consider_majority:
            most_common_color = color
            break  # Stop at the first one found
    
    # If no valid common color is found, use the RGB parameter
    if most_common_color is None:
        if RGB is None:
            raise ValueError("No common color found and no RGB parameter provided")
        target_color = tuple(RGB) + (255,)  # Include alpha (full opacity)
    else:
        target_color = most_common_color
    
    # Create a new list to store the modified pixel data
    new_data = []
    
    for pixel in data:
        # If the pixel is the most common color, set it to transparent
        if pixel == target_color:
            new_data.append((0, 0, 0, 0))  # Transparent pixel (R, G, B, A)
        else:
            new_data.append(pixel)  # Keep the original pixel

    # Update the image with new pixel data
    img.putdata(new_data)
    
    # Save the modified image to a BytesIO buffer
    output_buffer = io.BytesIO()
    img.save(output_buffer, format="PNG")
    output_buffer.seek(0)  # Move the cursor back to the start of the buffer
    
    return output_buffer, target_color

def main():
    st.title('Transparent LaTeX renderer')

    # User input for LaTeX code
    latex_input = st.text_area('Enter your LaTeX code:', r'\int_0^\infty e^{-x^2} dx')
    
    # User input for color
    latex_color = st.color_picker('Pick a color for the LaTeX text', '#000000')  # Default black

    # Render LaTeX as text
    # if latex_input:
    #     st.latex(latex_input)

    
    if latex_input:
        # Render the LaTeX expression to an image
        img = render_latex_to_image(latex_input, latex_color)

        # Process the image to make certain pixels transparent
        transparent_img_io, background_rgba = make_color_pixels_transparent(img)

        # Convert BytesIO back to a PIL image
        transparent_img = Image.open(transparent_img_io)

        # Convert transparent image to Base64
        buffer = io.BytesIO()
        transparent_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # HTML for displaying the transparent image over a background color
        html_code = f"""
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: rgb{background_rgba};  
            padding: 20px;
        ">
            <img src="data:image/png;base64,{img_base64}" 
                style="max-width: 100%; height: auto;" 
                alt="Rendered LaTeX Image"/>
        </div>
        """

        # Render the HTML with CSS in Streamlit
        st.markdown(html_code, unsafe_allow_html=True)
if __name__ == '__main__':
    main()
