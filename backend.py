from flask import Flask, request, make_response, jsonify
import os

supported_types = ['jpg', 'png'] 
app = Flask(__name__) 

app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/test', methods=['GET'])
def test():
    return 'Just a test!'

@app.route('/post_image', methods=['POST'])
def post_image():
    # POST 

    res = make_response(jsonify({"status": "FAIL", "msg": "No file uploaded"}), 400)
    return res

if __name__ == '__main__':
    app.run(debug=True)