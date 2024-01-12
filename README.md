# Chest X-ray Classification Web App

## Overview
This web application is designed to classify chest X-ray images into three categories: 'covid', 'normal', and 'pneumonia'. It uses a Convolutional Neural Network (CNN) built with TensorFlow and integrated into a Flask web application.

## Project Structure
The project follows a standard Flask directory structure with additional directories and files:

- **Static Directory:**
  - **css:** Stylesheets for styling HTML templates.
  - **fonts:** Font files.
  - **images:** Static images.

- **Templates Directory:**
  - **about.html:** About page template.
  - **index.html:** Main/home page template.
  - **main.html:** Another main or generic template.
  - **symptoms.html:** Template for a page related to symptoms.

- **Files:**
  - **prepros-6.config:** Configuration file for Prepros.
  - **readme.bit:** README file containing project information and instructions.

- **Python Script:**
  - **app.py:** Main Flask application script containing routes, views, and logic.

- **Model File:**
  - **custom_cnn_model.h5:** Saved Keras model file for chest X-ray classification.
## Technologies Used
- **Python:** Programming language used for the backend logic. ![Python Icon](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
- **Flask:** Web framework used for building the web application. ![Flask Icon](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
- **TensorFlow:** Machine learning library used for implementing the Convolutional Neural Network. ![TensorFlow Icon](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
## Setup Instructions
1. Clone the repository: `git clone https://github.com/your-username/your-repo.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask application: `python app.py`
4. Access the web app in your browser: `http://localhost:5000`

## Contributors
- [Zeineb Zaiet](https://github.com/your-username)
- [Sana Khamassi](https://github.com/Sanakhamassi)
- [Nada Ben taarit](https://github.com/contributor1)
- [Feriel Ben Rjeeb](https://github.com/contributor2)


## License
This project is licensed under the [MIT License](LICENSE).
