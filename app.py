import os
import tempfile
from flask import Flask, render_template, request, jsonify, send_file, session
import database
import data_loader
import rag
import auth
import audio

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ====================================================================
# KHỞI TẠO VÀ CẤU HÌNH CÁC MÔ HÌNH VÀ HỆ THỐNG
# ====================================================================

conn = database.set_connection()
if conn:
    products = data_loader.load_products(conn)
    key_attribute = data_loader.get_unique_attribute_keys(products)
    search_engine = data_loader.load_semantic_search_engine(products)
    whisper_model = data_loader.load_stt_model()
    qwen_api_client = data_loader.get_qwen_api_client() 

    rag.initialize_rag(products, search_engine, qwen_api_client, key_attribute)
    
# ====================================================================
# CÁC ROUTE API CỦA FLASK
# ====================================================================

@app.route('/')
def index():
    """Route chính để hiển thị trang HTML."""
    return render_template('index.html')

@app.route('/stt', methods=['POST'])
def speech_to_text():
    """API để chuyển giọng nói thành văn bản (Speech-to-Text)."""
    if not whisper_model:
        return jsonify({'error': 'Mô hình Whisper chưa được tải.'}), 503

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Không có file audio được gửi lên.'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Không có file được chọn.'}), 400

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                audio_file.save(tmp_file.name)
                filepath = tmp_file.name
            
            print(f"Đang xử lý STT cho file: {filepath}")
            wav_path = None
            chunks = []
            try:
                wav_path = audio.convert_to_wav(filepath)
                if not wav_path:
                    return jsonify({'error': 'Không thể chuyển đổi định dạng âm thanh.'}), 500

                chunks = audio.split_audio_chunks(wav_path, data_loader.CHUNK_LENGTH_MS)
                transcript = audio.transcribe_audio_chunks(whisper_model, chunks)
                print(f"Kết quả STT: {transcript}")
                return jsonify({'transcript': transcript})
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
                for chunk_path in chunks:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

    except Exception as e:
        print(f"Lỗi STT: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """API để chuyển văn bản thành giọng nói (Text-to-Speech)."""
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Không có văn bản để chuyển đổi.'}), 400

        print(f"Đang xử lý TTS cho văn bản: '{text}'")
        audio_stream = audio.text_to_speech_gtts(text)
        
        print("Đã tạo audio thành công, đang gửi về client.")
        return send_file(audio_stream, mimetype='audio/mpeg') 

    except Exception as e:
        print(f"Lỗi TTS: {e}") 
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint chính để xử lý tin nhắn và tạo câu trả lời."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "Chưa đăng nhập."}), 401

    data = request.json
    message = data.get('message')
    session_id = data.get('sessionId')

    if not message:
        return jsonify({"status": "error", "message": "Tin nhắn không được để trống."}), 400

    if not session_id:
        new_session = database.create_session(user_id)
        session_id = new_session['ma_phien']
        database.save_message(session_id, 'user', message)
        database.update_session_title(session_id, message.split()[0].title() if len(message) > 1 else message)
    else:
        database.save_message(session_id, 'user', message)

    messages = database.get_session_messages(session_id, limit=4)
    conversation_ctx = [] 
    if len(messages) % 2 != 0:
        messages.pop(0)

    for i in range(0, len(messages), 2):
        if i + 1 < len(messages):
            user_msg = messages[i].get('noi_dung', '')
            bot_msg = messages[i+1].get('noi_dung', '')
            conversation_ctx.append(f"User: {user_msg}\nBot: {bot_msg}")
    print("DEBUG conversation_ctx =", conversation_ctx)
    
    # Truyền conversation_ctx vào hàm answer_query
    bot_response = rag.answer_query(message, conversation_ctx)
    
    database.save_message(session_id, 'bot', bot_response)
    return jsonify({"status": "success", "response": bot_response, "sessionId": session_id})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Lấy danh sách các phiên trò chuyện của người dùng hiện tại."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "Chưa đăng nhập."}), 401
    
    sessions = database.get_sessions_by_user(user_id)
    return jsonify({"status": "success", "data": sessions})

@app.route('/api/sessions/<session_id>/messages', methods=['GET'])
def get_messages(session_id):
    """Lấy tin nhắn của một phiên trò chuyện cụ thể."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "Chưa đăng nhập."}), 401
        
    messages = database.get_messages_by_session(session_id)
    return jsonify({"status": "success", "messages": messages})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def api_delete_session(session_id):
    """Xóa một phiên trò chuyện."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"status": "error", "message": "Chưa đăng nhập."}), 401
    
    sessions_by_user = database.get_sessions_by_user(user_id)
    session_to_delete = next((s for s in sessions_by_user if s['ma_phien'] == session_id), None)
    if not session_to_delete:
        return jsonify({"status": "error", "message": "Bạn không có quyền xóa phiên này."}), 403

    result = database.delete_session(session_id)
    return jsonify(result)

@app.route('/api/register', methods=['POST'])
def api_register():
    """Endpoint API để đăng ký người dùng mới."""
    data = request.json
    ho_ten = data.get('ho_ten')
    email = data.get('email')
    ten_dang_nhap = data.get('ten_dang_nhap')
    mat_khau = data.get('mat_khau')

    if not all([ho_ten, email, ten_dang_nhap, mat_khau]):
        return jsonify({"status": "error", "message": "Vui lòng điền đầy đủ thông tin."}), 400

    result = auth.register_user(ho_ten, email, ten_dang_nhap, mat_khau)
    return jsonify(result)

@app.route('/api/login', methods=['POST'])
def api_login():
    """Endpoint API để đăng nhập người dùng."""
    data = request.json
    ten_dang_nhap = data.get('ten_dang_nhap')
    mat_khau = data.get('mat_khau')

    if not ten_dang_nhap or not mat_khau:
        return jsonify({"status": "error", "message": "Vui lòng nhập tên đăng nhập và mật khẩu."}), 400

    result = auth.login_user(ten_dang_nhap, mat_khau)
    if result['status'] == 'success':
        session['user_id'] = result['user']['ma_nguoi_dung']
        return jsonify(result)
    else:
        return jsonify(result), 401

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Xử lý đăng xuất người dùng."""
    session.pop('user_id', None)
    return jsonify({"status": "success", "message": "Đã đăng xuất."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7890, debug=True)