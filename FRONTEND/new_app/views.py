from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# ================= CUSTOM LAYER =================
class CastLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

# ================= LOAD MODEL =================
with custom_object_scope({'CastLayer': CastLayer, 'Cast': CastLayer}):
    model = load_model(
        r"C:\Users\molak\Music\Brain Tumour CNN\FRONTEND\brain_tumor_CNN_.h5",
        safe_mode=False
    )

# ================= CLASS LABELS (MATCH TRAINING ORDER) =================
class_names = [
    'glioma_tumor',
    'meningioma_tumor',
    'no_tumor',
    'pituitary_tumor'
]

# ================= TUMOR INFO DICTIONARY =================
TUMOR_INFO = {
    'pituitary_tumor': {
        'display_name': 'Pituitary Tumor',
        'color_class': 'pituitary',
        'description': (
            'Pituitary tumors develop in the pituitary gland at the base of the brain. '
            'Most pituitary tumors are benign (noncancerous) and are called pituitary adenomas. '
            'They can affect hormone production, leading to a variety of symptoms depending on '
            'which hormones are involved.'
        ),
        'next_steps': [
            'Consult an endocrinologist for hormonal evaluation and blood tests.',
            'Schedule a high-resolution contrast-enhanced MRI for detailed assessment.',
            'Discuss treatment options — many pituitary tumors respond well to medication.',
            'Monitor vision regularly, as pituitary tumors can compress the optic nerves.',
        ],
    },
    'glioma_tumor': {
        'display_name': 'Glioma Tumor',
        'color_class': 'glioma',
        'description': (
            'Gliomas originate from glial cells that surround and support neurons in the brain. '
            'They are the most common type of primary brain tumor and can vary widely in aggressiveness, '
            'from low-grade (slow-growing) to high-grade (fast-growing glioblastomas). '
            'Early detection and grading are critical for treatment planning.'
        ),
        'next_steps': [
            'Seek an urgent consultation with a neuro-oncologist for tumor grading.',
            'Undergo advanced imaging (MRI with spectroscopy or PET scan) for precise characterization.',
            'Discuss biopsy or surgical options to confirm the histological grade.',
            'Explore treatment plans including surgery, radiation therapy, and chemotherapy.',
        ],
    },
    'meningioma_tumor': {
        'display_name': 'Meningioma Tumor',
        'color_class': 'meningioma',
        'description': (
            'Meningiomas arise from the meninges — the protective membranes surrounding the brain '
            'and spinal cord. They are typically slow-growing and benign in most cases (WHO Grade I). '
            'However, their location can cause significant neurological symptoms by compressing '
            'adjacent brain tissue.'
        ),
        'next_steps': [
            'Consult a neurosurgeon for assessment of tumor size and location.',
            'Schedule periodic MRI follow-ups to monitor growth rate.',
            'Discuss surgical resection if the tumor is symptomatic or growing.',
            'Consider stereotactic radiosurgery as a non-invasive alternative for small tumors.',
        ],
    },
    'no_tumor': {
        'display_name': 'No Tumor Detected',
        'color_class': 'no_tumor',
        'description': (
            'The CNN model did not detect any signs of a brain tumor in this MRI scan. '
            'The scan appears to show normal brain anatomy without visible masses or abnormalities. '
            'This is a positive preliminary result, but should always be confirmed by a qualified '
            'radiologist for clinical decisions.'
        ),
        'next_steps': [
            'Share this result with your physician for confirmation and medical records.',
            'Continue routine health check-ups and screenings as recommended.',
            'If symptoms persist, consider a follow-up scan or second opinion.',
            'Maintain a healthy lifestyle to support long-term brain health.',
        ],
    },
}


# ================= IMAGE PREPROCESSING =================
def preprocess_image(image):
    img = Image.open(image).convert('RGB')

    #  MUST MATCH TRAINING SIZE
    img = img.resize((124, 124))

    img_array = np.array(img)  # No /255.0 — training used raw 0-255 values
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ================= HOME =================
def home(request):
    return render(request, 'index.html')

# ================= LOGIN =================
def input(request):
    file_name = 'account.txt'
    name = request.POST.get('name')
    password = request.POST.get('password')

    with open(file_name, 'r') as file:
        accounts = [line.split() for line in file]

    for acc in accounts:
        if acc[0] == name and acc[1] == password:
            return render(request, 'input.html')

    return HttpResponse("Wrong Username or Password")

# ================= OUTPUT / PREDICTION =================
def output(request):
    if 'file' not in request.FILES:
        return HttpResponse("No file uploaded")

    img = request.FILES['file']
    processed_img = preprocess_image(img)

    predictions = model.predict(processed_img)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Get tumor info for the template
    info = TUMOR_INFO.get(predicted_class, TUMOR_INFO['no_tumor'])

    return render(request, 'output.html', {
        'out': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'confidence_val': f"{confidence:.1f}",
        'display_name': info['display_name'],
        'color_class': info['color_class'],
        'description': info['description'],
        'next_steps': info['next_steps'],
    })
