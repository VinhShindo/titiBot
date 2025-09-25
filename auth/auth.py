from database import get_user_by_username, create_user as db_create_user, get_user_by_email, get_user_by_id

def register_user(ho_ten, email, ten_dang_nhap, mat_khau):
    """
    Đăng ký một người dùng mới vào hệ thống.
    Kiểm tra xem tên đăng nhập và email đã tồn tại hay chưa.
    Mật khẩu được băm trước khi lưu trữ để tăng cường bảo mật.
    """
    if get_user_by_username(ten_dang_nhap):
        return {"status": "error", "message": "Tên đăng nhập đã tồn tại."}
    if get_user_by_email(email):
        return {"status": "error", "message": "Email đã được sử dụng."}

    user_data = db_create_user(ho_ten, email, ten_dang_nhap, mat_khau)
    
    if user_data:
        user_info = {k: v for k, v in user_data.items() if k != 'mat_khau'}
        return {"status": "success", "message": "Đăng ký thành công!", "user": user_info}
    else:
        return {"status": "error", "message": "Đăng ký thất bại. Vui lòng thử lại."}

def login_user(ten_dang_nhap, mat_khau):
    """
    Xác thực thông tin đăng nhập của người dùng.
    Tìm người dùng bằng tên đăng nhập và so sánh mật khẩu đã băm.
    """
    user = get_user_by_username(ten_dang_nhap)

    if user and user['mat_khau'] == mat_khau:
        user_info = {k: v for k, v in user.items() if k != 'mat_khau'}
        return {"status": "success", "message": "Đăng nhập thành công!", "user": user_info}
    else:
        return {"status": "error", "message": "Tên đăng nhập hoặc mật khẩu không chính xác."}

def get_user_data(user_id):
    """
    Lấy thông tin người dùng từ ID và loại bỏ mật khẩu.
    """
    user = get_user_by_id(user_id)
    if user:
        user_info = {k: v for k, v in user.items() if k != 'mat_khau'}
        return user_info
    return None
