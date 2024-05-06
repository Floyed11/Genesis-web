from flask import request
from flask import Blueprint
from flask import jsonify, render_template, redirect, url_for
from be.model.trans import Trans

bp_trans = Blueprint("trans", __name__, url_prefix="/trans")


@bp_trans.route("/trans_in", methods=["GET", "POST"])
def trans_in():
    if request.method == "GET":
        return render_template('input.html')
    elif request.method == "POST":
        data: dict = request.json
        t = Trans()
        # print(data)
        code, message, score = t.trans_in(data)
        return jsonify({'redirect': url_for('trans.get_result')})
    

@bp_trans.route("/get_result", methods=["GET", "POST"])
def get_result():
    t = Trans()
    code, message, score = t.get_result()
    # print('score', score)
    # return jsonify({"message": message, "score": score}), code
    return render_template("output.html", score=score)