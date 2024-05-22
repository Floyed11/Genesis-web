from flask import request
from flask import Blueprint
from flask import jsonify, render_template, redirect, url_for, send_from_directory
from be.model.trans import Trans
from be.model.data import train_new_model

bp_trans = Blueprint("trans", __name__, url_prefix="/trans")

max_mhz = 0
nominal = 0
base_threads = 0
enabled_chips = 0
enabled_cores = 0
threads_core = 0


@bp_trans.route("/", methods=["GET", "POST"])
def index():
    return redirect(url_for('trans.trans_in'))

@bp_trans.route("/trans_in", methods=["GET", "POST"])
def trans_in():
    if request.method == "GET":
        return render_template('input.html')
    elif request.method == "POST":
        data: dict = request.json
        t = Trans()
        # print(data)
        global max_mhz, nominal, base_threads, enabled_chips, enabled_cores, threads_core
        max_mhz = data['maxMHz']
        nominal = data['nominal']
        base_threads = data['baseThreads']
        enabled_chips = data['enabledChips']
        enabled_cores = data['enabledCores']
        threads_core = data['threadsCore']
        code, message, score = t.trans_in(data)
        print('score', score)
        return jsonify({'redirect': url_for('trans.result')})
    

@bp_trans.route("/get_result", methods=["GET", "POST"])
def get_result():
    t = Trans()
    code, message, score = t.get_result()
    # print('score', score)
    # return jsonify({"message": message, "score": score}), code
    return render_template("output.html", score=score)


@bp_trans.route("/train_new", methods=["GET", "POST"])
def train_new():
    data = request.get_json()
    iterations = data.get('iterations', "500000")
    train_new_model(iterations)
    return redirect(url_for('trans.trans_in'))


@bp_trans.route("/result_view", methods=["GET", "POST"])
def result_view():
    return render_template('result_view.html')


@bp_trans.route('/static/<path:filename>')
def dataset(filename):
    return send_from_directory(bp_trans.static_folder, filename)


@bp_trans.route("/result", methods=["GET", "POST"])
def result():
    global max_mhz, nominal, base_threads, enabled_chips, enabled_cores, threads_core
    t = Trans()
    code, message, score = t.get_result()
    return render_template('result_result.html', score=score, max_mhz=max_mhz, nominal=nominal, base_threads=base_threads, enabled_chips=enabled_chips, enabled_cores=enabled_cores, threads_core=threads_core)