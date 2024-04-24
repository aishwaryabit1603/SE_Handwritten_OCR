{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e9d7c1e0-7337-466b-94b0-33888234377d",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting streamlit\n",
      "  Obtaining dependency information for streamlit from https://files.pythonhosted.org/packages/b7/f9/9ad1e6e33e4ae149aead1ee8062e536d060b50d650df710899805562910a/streamlit-1.33.0-py2.py3-none-any.whl.metadata\n",
      "  Downloading streamlit-1.33.0-py2.py3-none-any.whl.metadata (8.5 kB)\n",
      "Collecting altair<6,>=4.0 (from streamlit)\n",
      "  Obtaining dependency information for altair<6,>=4.0 from https://files.pythonhosted.org/packages/46/30/2118537233fa72c1d91a81f5908a7e843a6601ccc68b76838ebc4951505f/altair-5.3.0-py3-none-any.whl.metadata\n",
      "  Downloading altair-5.3.0-py3-none-any.whl.metadata (9.2 kB)\n",
      "Collecting blinker<2,>=1.0.0 (from streamlit)\n",
      "  Obtaining dependency information for blinker<2,>=1.0.0 from https://files.pythonhosted.org/packages/fa/2a/7f3714cbc6356a0efec525ce7a0613d581072ed6eb53eb7b9754f33db807/blinker-1.7.0-py3-none-any.whl.metadata\n",
      "  Downloading blinker-1.7.0-py3-none-any.whl.metadata (1.9 kB)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (5.3.2)\n",
      "Requirement already satisfied: click<9,>=7.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (8.0.4)\n",
      "Requirement already satisfied: numpy<2,>=1.19.3 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (1.24.3)\n",
      "Requirement already satisfied: packaging<25,>=16.8 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (23.1)\n",
      "Requirement already satisfied: pandas<3,>=1.3.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (2.0.3)\n",
      "Requirement already satisfied: pillow<11,>=7.1.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (10.0.1)\n",
      "Requirement already satisfied: protobuf<5,>=3.20 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (4.25.2)\n",
      "Requirement already satisfied: pyarrow>=7.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (11.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (2.31.0)\n",
      "Collecting rich<14,>=10.14.0 (from streamlit)\n",
      "  Obtaining dependency information for rich<14,>=10.14.0 from https://files.pythonhosted.org/packages/87/67/a37f6214d0e9fe57f6ae54b2956d550ca8365857f42a1ce0392bb21d9410/rich-13.7.1-py3-none-any.whl.metadata\n",
      "  Downloading rich-13.7.1-py3-none-any.whl.metadata (18 kB)\n",
      "Requirement already satisfied: tenacity<9,>=8.1.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (8.2.2)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (4.7.1)\n",
      "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
      "  Obtaining dependency information for gitpython!=3.1.19,<4,>=3.0.7 from https://files.pythonhosted.org/packages/e9/bd/cc3a402a6439c15c3d4294333e13042b915bbeab54edc457c723931fed3f/GitPython-3.1.43-py3-none-any.whl.metadata\n",
      "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
      "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
      "  Obtaining dependency information for pydeck<1,>=0.8.0b4 from https://files.pythonhosted.org/packages/57/9f/0f701126356a1b2f6a5513dd96362c1f3c202391943a69b7fc95f93b44a9/pydeck-0.8.1b1-py2.py3-none-any.whl.metadata\n",
      "  Downloading pydeck-0.8.1b1-py2.py3-none-any.whl.metadata (3.9 kB)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from streamlit) (6.3.2)\n",
      "Requirement already satisfied: jinja2 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (3.1.2)\n",
      "Requirement already satisfied: jsonschema>=3.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (4.17.3)\n",
      "Requirement already satisfied: toolz in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from altair<6,>=4.0->streamlit) (0.12.0)\n",
      "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Obtaining dependency information for gitdb<5,>=4.0.1 from https://files.pythonhosted.org/packages/fd/5b/8f0c4a5bb9fd491c277c21eff7ccae71b47d43c4446c9d0c6cff2fe8c2c4/gitdb-4.0.11-py3-none-any.whl.metadata\n",
      "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3.post1)\n",
      "Requirement already satisfied: tzdata>=2022.1 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from pandas<3,>=1.3.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (1.26.16)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from requests<3,>=2.27->streamlit) (2023.11.17)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
      "  Obtaining dependency information for smmap<6,>=3.0.1 from https://files.pythonhosted.org/packages/a7/a5/10f97f73544edcdef54409f1d839f6049a0d79df68adbc1ceb24d1aaca42/smmap-5.0.1-py3-none-any.whl.metadata\n",
      "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.1)\n",
      "Requirement already satisfied: attrs>=17.4.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (22.1.0)\n",
      "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.18.0)\n",
      "Requirement already satisfied: mdurl~=0.1 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
      "Downloading streamlit-1.33.0-py2.py3-none-any.whl (8.1 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.1/8.1 MB\u001b[0m \u001b[31m24.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0ma \u001b[36m0:00:01\u001b[0m\n",
      "\u001b[?25hDownloading altair-5.3.0-py3-none-any.whl (857 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m857.8/857.8 kB\u001b[0m \u001b[31m31.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hDownloading blinker-1.7.0-py3-none-any.whl (13 kB)\n",
      "Downloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m25.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hDownloading pydeck-0.8.1b1-py2.py3-none-any.whl (7.0 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m7.0/7.0 MB\u001b[0m \u001b[31m27.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25hDownloading rich-13.7.1-py3-none-any.whl (240 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m240.7/240.7 kB\u001b[0m \u001b[31m11.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hDownloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
      "\u001b[?25hDownloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
      "Installing collected packages: smmap, blinker, rich, pydeck, gitdb, gitpython, altair, streamlit\n",
      "Successfully installed altair-5.3.0 blinker-1.7.0 gitdb-4.0.11 gitpython-3.1.43 pydeck-0.8.1b1 rich-13.7.1 smmap-5.0.1 streamlit-1.33.0\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "1e9698ab-10cf-4575-8fce-c429916895fa",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting opencv-python-headless\n",
      "  Obtaining dependency information for opencv-python-headless from https://files.pythonhosted.org/packages/32/0c/a59f2a40d6058ee8126668dc5dff6977c913f6ecd21dbd15b41563409a18/opencv_python_headless-4.9.0.80-cp37-abi3-macosx_11_0_arm64.whl.metadata\n",
      "  Downloading opencv_python_headless-4.9.0.80-cp37-abi3-macosx_11_0_arm64.whl.metadata (20 kB)\n",
      "Requirement already satisfied: numpy>=1.21.2 in /Users/anilkumar/anaconda3/lib/python3.11/site-packages (from opencv-python-headless) (1.24.3)\n",
      "Downloading opencv_python_headless-4.9.0.80-cp37-abi3-macosx_11_0_arm64.whl (35.4 MB)\n",
      "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m35.4/35.4 MB\u001b[0m \u001b[31m15.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m00:01\u001b[0m00:01\u001b[0m\n",
      "\u001b[?25hInstalling collected packages: opencv-python-headless\n",
      "Successfully installed opencv-python-headless-4.9.0.80\n"
     ]
    }
   ],
   "source": [
    "!pip install opencv-python-headless"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "537ce3be-4962-4acf-b5d1-a614bb371d33",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from io import StringIO\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "7b1bc03e-439b-46c1-ac93-46b862ddaea5",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "st.title(\"Handwritten Character Recoginition\")\n",
    "st.write(\"Handwritten Character Recoginition\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "cb253557-af6d-46c0-af21-cedfc9cb9147",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#load model, set cache to prevent reloading\n",
    "@st.cache(allow_output_mutation=True)\n",
    "def load_model():\n",
    "    model=tf.keras.models.load_model('Desktop/jupyter_lab/projects/ocr/DenseNet121_model.h5')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "3c39f208-521d-44ad-b2b0-8461aa2133b7",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# load image\n",
    "# Convert image to grayscale and resize\n",
    "def load_image(image, target_size=(64, 64)):\n",
    "    img = tf.image.decode_jpeg(image, channels=3)\n",
    "    img = tf.cast(img, tf.float32)\n",
    "    img /= 255.0\n",
    "    img = tf.image.resize(img, target_size)\n",
    "    img = tf.expand_dims(img, axis=0)\n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "688e0788-337f-4561-bed1-b9f143fbc05a",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Binarization of the grayscale image\n",
    "def binarization(image):\n",
    "    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)\n",
    "    return thresh\n",
    "\n",
    "# Dilate the image to enhance character recognition\n",
    "def dilate(image, words=False):\n",
    "    kernel_size = (6, 6) if words else (3, 3)\n",
    "    iterations = 3 if words else 4\n",
    "    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)\n",
    "    dilated_image = cv2.dilate(image, kernel, iterations=iterations)\n",
    "    return dilated_image\n",
    "\n",
    "# Find contours and rectangles in the image\n",
    "def find_rectangles(image):\n",
    "    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)\n",
    "    rectangles = [cv2.boundingRect(cnt) for cnt in contours]\n",
    "    sorted_rectangles = sorted(rectangles, key=lambda x: x[0])\n",
    "    return sorted_rectangles\n",
    "\n",
    "# Extract characters from the image\n",
    "def extract_characters(image):\n",
    "    model = load_model()\n",
    "    chars = []\n",
    "\n",
    "    resized_image = cv2.resize(image, (256, 256))  # Resize image to standard size\n",
    "    processed_image = preprocess_image(resized_image)\n",
    "    binarized_image = binarization(processed_image)\n",
    "    dilated_image = dilate(binarized_image, words=True)\n",
    "    words = find_rectangles(dilated_image)\n",
    "\n",
    "    for word in words:\n",
    "        x, y, w, h = word\n",
    "        img = image[y:y+h, x:x+w]\n",
    "\n",
    "        binarized_word = binarization(preprocess_image(img))\n",
    "        dilated_word = dilate(binarized_word)\n",
    "        characters = find_rectangles(dilated_word)\n",
    "\n",
    "        for char in characters:\n",
    "            x, y, w, h = char\n",
    "            ch = img[y:y+h, x:x+w]\n",
    "\n",
    "            # Perform further processing or recognition on the character 'ch'\n",
    "\n",
    "    del model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "10a322b6-379d-4432-b281-fd2f0151b220",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Main function to create the Streamlit app\n",
    "def main():\n",
    "    st.title(\"Handwritten Character Recognition\")\n",
    "    uploaded_file = st.file_uploader(\"Upload Image\", type=['jpg', 'jpeg', 'png'])\n",
    "\n",
    "    if uploaded_file:\n",
    "        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)\n",
    "        st.image(image, caption='Uploaded Image', use_column_width=True)\n",
    "\n",
    "        if st.button('Recognize Characters'):\n",
    "            extracted_text = extract_characters(image)\n",
    "            st.write('Extracted Text:', extracted_text)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
