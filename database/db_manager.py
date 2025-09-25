from supabase import create_client, Client
import os

# ==========================
# 1. KẾT NỐI SUPABASE
# ==========================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://sywyxrvfofmqwrfqfsws.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InN5d3l4cnZmb2ZtcXdyZnFmc3dzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjQ3MzY1MiwiZXhwIjoyMDcyMDQ5NjUyfQ.TFNiNDome9-R179KmKM5NEfZoNUwNKe_ewYN9qlGiw4")  # đổi key cho đúng

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def set_connection():
    return supabase

# ==========================
# 2. HÀM HỖ TRỢ
# ==========================
def _generate_next_id(table_name: str, id_column_name: str, id_prefix: str) -> str:
    """
    Sinh ID tuần tự: vd "nd0001", "nd0002"...
    """
    response = supabase.table(table_name).select(id_column_name).like(id_column_name, f"{id_prefix}%").order(id_column_name, desc=True).limit(1).execute()
    last_id = response.data[0][id_column_name] if response.data else None

    if last_id:
        last_id_num_str = last_id[len(id_prefix):]
        if last_id_num_str.isdigit():
            next_num = int(last_id_num_str) + 1
        else:
            next_num = 1
    else:
        next_num = 1

    return f"{id_prefix}{next_num:04d}"


# ==========================
# 3. NGƯỜI DÙNG
# ==========================
def create_user(ho_ten, email, ten_dang_nhap, hashed_password):
    new_id = _generate_next_id('nguoidung', 'ma_nguoi_dung', 'nd')
    data = {
        "ma_nguoi_dung": new_id,
        "ho_ten": ho_ten,
        "email": email,
        "ten_dang_nhap": ten_dang_nhap,
        "mat_khau": hashed_password
    }
    response = supabase.table("nguoidung").insert(data).execute()
    return response.data[0] if response.data else None


def get_user_by_username(ten_dang_nhap):
    response = supabase.table("nguoidung").select("*").eq("ten_dang_nhap", ten_dang_nhap).execute()
    return response.data[0] if response.data else None


def get_user_by_email(email):
    response = supabase.table("nguoidung").select("*").eq("email", email).execute()
    return response.data[0] if response.data else None


def get_user_by_id(user_id):
    response = supabase.table("nguoidung").select("*").eq("ma_nguoi_dung", user_id).execute()
    return response.data[0] if response.data else None


# ==========================
# 4. PHIÊN TRÒ CHUYỆN
# ==========================
def create_session(ma_nguoi_dung):
    new_id = _generate_next_id('phientrochuyen', 'ma_phien', 'ph')
    data = {
        "ma_phien": new_id,
        "ma_nguoi_dung": ma_nguoi_dung,
        "tieu_de": "Cuộc trò chuyện mới"
    }
    response = supabase.table("phientrochuyen").insert(data).execute()
    return response.data[0] if response.data else None


def get_sessions_by_user(ma_nguoi_dung):
    response = supabase.table("phientrochuyen").select("*").eq("ma_nguoi_dung", ma_nguoi_dung).order("bat_dau_luc", desc=True).execute()
    return response.data


def update_session_title(ma_phien, tieu_de):
    supabase.table("phientrochuyen").update({"tieu_de": tieu_de}).eq("ma_phien", ma_phien).execute()


def delete_session(ma_phien):
    supabase.table("phientrochuyen").delete().eq("ma_phien", ma_phien).execute()


# ==========================
# 5. TIN NHẮN
# ==========================
def save_message(ma_phien, nguoi_gui, noi_dung):
    new_id = _generate_next_id('tinnhan', 'ma_tin_nhan', 'tn')
    data = {
        "ma_tin_nhan": new_id,
        "ma_phien": ma_phien,
        "nguoi_gui": nguoi_gui,
        "noi_dung": noi_dung
    }
    response = supabase.table("tinnhan").insert(data).execute()
    return response.data[0] if response.data else None


def get_messages_by_session(ma_phien):
    response = supabase.table("tinnhan").select("*").eq("ma_phien", ma_phien).order("tao_luc").execute()
    return response.data


def get_session_messages(ma_phien, limit: int = 4):
    response = (
        supabase.table("tinnhan")
        .select("ma_tin_nhan, nguoi_gui, noi_dung, tao_luc")
        .eq("ma_phien", ma_phien)
        .order("tao_luc")
        .limit(limit)
        .execute()
    )
    return response.data
