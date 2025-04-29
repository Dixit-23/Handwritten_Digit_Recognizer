from fastai.vision.all import * 
import gradio as gr

learn = load_learner('Handwritten_Digit_Recognition.pkl')

categories = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
def classify_image(img) :
    pred, idx, probs = learn. predict(img)
    return dict(zip(categories, map(float, probs) ))

image = gr.Image()
label = gr.Label()
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label)
intf.launch(inline=False,share=True)
