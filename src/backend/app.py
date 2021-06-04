from flask import Flask, request, make_response, jsonify
import os

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return "Neural Style Transfer"

@app.route('/test', methods=['GET'])
def test_get():
    return 'Just a test!'

@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    content_file = request.files['content_file']
    style_file = request.files['style_file']

    if content_file and style_file:
        res = make_response(jsonify({"status": "SUCCESS", "msg": "No file uploaded"}), 200)
    else:
        res = make_response(jsonify({"status": "FAIL", "msg": "No file uploaded"}), 400)
    return res

if __name__ == '__main__':
    app.run(debug=True)