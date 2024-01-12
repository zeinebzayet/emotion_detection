#from CNN_algorithm.CNN import build_and_train_model, classify_an_image, performance, pre_process, predict
from CNN_algorithm.CNN import build_and_train_model, classify_an_image, performance, pre_process, predict

from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)
model = None
@app.route('/home',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/about',methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/symptoms',methods=['GET'])
def traitement():
    return render_template('emotion.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global model
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save('static/images/img.png')  # Save the uploaded file
            if model is None:
            # Perform prediction after file upload
                x_test, y_test, y_test_lb, x_train, y_train, y_train_lb = pre_process()
                model = build_and_train_model(x_train, y_train_lb, x_test, y_test_lb)
                performance(model, x_test, y_test_lb)
                predict(model, x_test, y_test)

            predicted_class = classify_an_image('static/images/img.png', model)
            print("Predicted Class:", predicted_class)
            return render_template('emotion.html', predicted_class=predicted_class, image_path='./static/images/img.png')
        else:
            # Handle cases where no file was uploaded
            return render_template('error.html')



if __name__ == '__main__':
    app.run(debug=True)
