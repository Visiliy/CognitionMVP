from flask import Flask, Response, request, render_template
from search_pipline import CognitionSearch
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('main.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        files = request.files.getlist("files")
        message = request.form.get("message", "")
        print("Сообщение:", message)
        for file in files:
            if file.filename == '':
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

        def generate():
            search = CognitionSearch("files", "mps")
            for chunk in search.generate_answer(message):
                yield chunk

        return Response(generate(), mimetype='text/plain')
    
    except Exception as e:
        print(e)
        return Response("Server error", status=502, mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True, port=5077)