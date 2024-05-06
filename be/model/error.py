error_code = {
    401: "authorization fail.",
    511: "non exist user id {}",
    512: "exist user id {}",
}


def error_non_exist_user_id(user_id):
    return 511, error_code[511].format(user_id)


def error_exist_user_id(user_id):
    return 512, error_code[512].format(user_id)


def error_wrong_state(order_id):
    return 520, error_code[520].format(order_id)


def error_authorization_fail():
    return 401, error_code[401]


def error_and_message(code, message):
    return code, message
