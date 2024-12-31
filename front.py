import gradio as gr
from data_labeller import DataLabeller
from save_cropped_images import save_cropped_images
from detect_text import TextDetector
from split_image import ImageSegmenter
from PIL import Image

def crop_and_transcribe(img_path):
    data_labeller = DataLabeller()
    segments = data_labeller.predict_image(img_path)
    segmenter = ImageSegmenter(data_labeller)
    cropped_images = segmenter.segment_and_crop(img_path)
    save_cropped_images(cropped_images,'./data')
    text_detector = TextDetector()
    transcribed_texts = text_detector.detect_texts('./data')
    return transcribed_texts

def process_image(image: Image.Image) -> str:
    # Convert the PIL image to a format suitable for your function
    img_path = "temp_image.jpg"
    image.save(img_path)
    result = crop_and_transcribe(img_path)
    formatted_result = ""
    for key, value in result.items():
        value = value.replace('\n', ' ')
        formatted_line = f"{key.split('.')[0]}: {value}"
        formatted_result += formatted_line + "\n"

    
    # formatted_result = "\n".join(f"{key.split('.')[0]}: {value.replace('\\n', ' ')}" for key, value in result.items())
    return formatted_result

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Processing Interface")
    
    # File upload for image
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")

    # Output text area
    output_text = gr.Textbox(label="Output", interactive=False)

    # Button to trigger processing
    process_button = gr.Button("Process Image")

    # Link the button to the function
    process_button.click(process_image, inputs=image_input, outputs=output_text)

# Launch the interface
demo.launch()
