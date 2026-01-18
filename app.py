import os
import json
from flask import Flask, Response, request, render_template, redirect, url_for
from search_pipline import CognitionSearch
from flask_cors import CORS
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_bcrypt import Bcrypt

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'super-secret-key')

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = 'user_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

users_db = {}
user_files = {}
user_chat_history = {}

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

    @staticmethod
    def get(user_id):
        return users_db.get(user_id)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/', methods=['GET'])
def index():
    return render_template('main.html', user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('main.html', user=current_user)
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    if not username or not password:
        return 'Ошибка: заполните все поля', 400
    
    if username in [u.username for u in users_db.values() if hasattr(u, 'username')]:
        return 'Пользователь уже существует', 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = str(len([k for k in users_db if not k.startswith('_')]) + 1)
    new_user = User(user_id, username, hashed_password)
    
    users_db[user_id] = new_user
    users_db[username] = new_user
    user_files[user_id] = []
    user_chat_history[user_id] = []
    
    return 'OK', 200

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'GET':
        return render_template('main.html', user=current_user)
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    user = users_db.get(username)
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        return 'OK', 200
    
    return 'Неверные данные', 400

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    user_id = current_user.id
    message = request.form.get("message", "").strip()
    
    if not message:
        return Response("Пустое сообщение", status=400, mimetype='text/plain')
    
    user_chat_history[user_id].append({"role": "user", "content": message})

    def generate():
        search = CognitionSearch(os.path.join(UPLOAD_FOLDER, user_id), "cpu")
        response_chunks = search.generate_answer(message, user_chat_history[user_id])
        full_response = ""
        for chunk in response_chunks:
            full_response += chunk
            yield chunk
        user_chat_history[user_id].append({"role": "assistant", "content": full_response})

    return Response(generate(), mimetype='text/plain; charset=utf-8')

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if 'files' not in request.files:
        return redirect(url_for('index'))
    
    files = request.files.getlist("files")
    user_id = current_user.id
    user_upload_folder = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(user_upload_folder, exist_ok=True)
    
    for file in files:
        if file.filename == '':
            continue
        filename = secure_filename(file.filename)
        if filename:
            filepath = os.path.join(user_upload_folder, filename)
            file.save(filepath)
            if user_id not in user_files:
                user_files[user_id] = []
            if filename not in user_files[user_id]:
                user_files[user_id].append(filename)
    
    return 'OK', 200

@app.route("/user_files", methods=["GET"])
@login_required
def get_user_files():
    user_id = current_user.id
    return json.dumps(user_files.get(user_id, []))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
