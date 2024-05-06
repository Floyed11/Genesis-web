from be import serve
from flask import Flask
from flask import render_template, redirect, url_for
from be.view import auth
from be.view import trans
from be.model.bench import init_database, init_completed_event
from flask_cors import CORS


if __name__ == "__main__":
    serve.be_run()


init_database()

app = Flask(__name__)
#app.config['WTF_CSRF_ENABLED'] = False
CORS(app)
app.register_blueprint(auth.bp_auth)
app.register_blueprint(trans.bp_trans)
init_completed_event.set()
app.run()

@app.route('/')
def index():
    return render_template('main.html')