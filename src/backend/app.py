from flask import Flask, request, make_response, jsonify
import os

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

from flask_cors import CORS

# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/')
def test():
    return "Neural Style Transfer v1"

@app.route('/test', methods=['GET'])
def test_get():
    return 'Just a test!'

@app.route('/post_images', methods=['POST'])
def post_images():
    # POST
    print('Post images')
    
    files = request.files.to_dict()

    if 'content_file' not in files or 'style_file' not in files:
        return make_response(jsonify({'status': 'ok'}), 200)

    content_file = files['content_file']
    style_file = request.files['style_file']

    res = make_response(jsonify({"status": "SUCCESS", "msg": "Files uploaded"}), 200)
    return res


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

