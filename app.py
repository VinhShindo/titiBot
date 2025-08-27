import os
import json
import tempfile
import base64
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import requests
from audio.stt_utils import transcribe_audio_chunks, convert_to_wav, split_audio_chunks
from audio.tts_utils import text_to_speech_gtts
from data_loader import whisper_model, CHUNK_LENGTH_MS
from rag import answer_query

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

MOCKAPI_URL = "https://68a67a43639c6a54e99ed73d.mockapi.io/NguoiDung"

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ====================================================================
# CÁC ROUTE API CỦA FLASK
# ====================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stt', methods=['POST'])
def speech_to_text():
    """API để chuyển giọng nói thành văn bản."""
    if not whisper_model:
        return jsonify({'error': 'Mô hình Whisper chưa được tải.'}), 503

    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Không có file audio được gửi lên.'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Không có file được chọn.'}), 400

        if audio_file:
            filename = audio_file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)

        print(f"Đang xử lý STT cho file: {filepath}")
        wav_path = None
        chunks = []
        try:
            wav_path = convert_to_wav(filepath)
            if not wav_path:
                return jsonify({'error': 'Không thể chuyển đổi định dạng âm thanh.'}), 500

            chunks = split_audio_chunks(wav_path, CHUNK_LENGTH_MS)
            transcript = transcribe_audio_chunks(whisper_model, chunks)
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
    """API để chuyển văn bản thành giọng nói."""
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Không có văn bản để chuyển đổi.'}), 400

        print(f"Đang xử lý TTS cho văn bản: '{text}'")
        audio_stream = text_to_speech_gtts(text)
        
        print("Đã tạo audio thành công, đang gửi về client.")
        return send_file(audio_stream, mimetype='audio/mpeg')

    except Exception as e:
        print(f"Lỗi TTS: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """API xử lý tin nhắn chat bằng cách sử dụng chức năng RAG."""
    try:
        # Lấy dữ liệu JSON từ request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Thiếu trường "question" trong request.'}), 400

        user_question = data['question']
        response = answer_query(user_question)

        return jsonify({'response': response})

    except Exception as e:
        # Xử lý các lỗi có thể xảy ra
        return jsonify({'error': str(e)}), 500

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    ho_ten = data.get('HoTen')
    email = data.get('Email')
    password = data.get('Password')

    if not ho_ten or not email or not password:
        return jsonify({"status": "error", "message": "Thiếu thông tin"}), 400

    res = requests.get(MOCKAPI_URL)
    if res.status_code != 200:
        return jsonify({"status": "error", "message": "Không kết nối được MockAPI"}), 500
    
    users = res.json()
    if any(u["Email"] == email for u in users):
        return jsonify({"status": "error", "message": "Email đã đăng ký"}), 400

    payload = {"HoTen": ho_ten, "Email": email, "MatKhau": password}
    res = requests.post(MOCKAPI_URL, json=payload)

    if res.status_code == 201:
        return jsonify({"status": "success", "message": "Đăng ký thành công"})
    else:
        return jsonify({"status": "error", "message": "Đăng ký thất bại"}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email = data.get('Email')
    password = data.get('Password')

    if not email or not password:
        return jsonify({"status": "error", "message": "Thiếu thông tin"}), 400

    res = requests.get(MOCKAPI_URL)
    if res.status_code != 200:
        return jsonify({"status": "error", "message": "Không kết nối được MockAPI"}), 500

    users = res.json()
    user = next((u for u in users if u["Email"] == email and u["MatKhau"] == password), None)

    if user:
        return jsonify({
            "status": "success",
            "message": f"Chào {user['HoTen']}",
            "user": {
                "id": user["maNguoiDung"],
                "name": user["HoTen"],
                "Email": user["Email"]
            }
        })
    else:
        return jsonify({"status": "error", "message": "Email hoặc mật khẩu không đúng"}), 401

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)