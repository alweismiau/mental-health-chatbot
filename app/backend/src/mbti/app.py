from flask import Flask, request, jsonify
from flask_cors import CORS
from mbti_test import calculate_mbti
from chatbot import chatbot_response, format_chat_history, summarize_chat
import requests

app = Flask(__name__)
CORS(app)

chat_histories = {}

# Endpoint MBTI Test
@app.route("/mbti-test", methods=["POST"])
def mbti_test():
    try:
        data = request.json
        answers = data.get("answers", [])

        if not answers or len(answers) != 40:
            return jsonify({"error": "Invalid input, must have 40 answers"}), 400

        result = calculate_mbti(answers)
        return jsonify({"mbti_result": result})  

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint Chat
# @app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("Data received:", data)

        user_input = data.get('user_input')
        mbti_result = data.get('mbti_result')
        user_id = data.get('user_id')
        chat_id = data.get('chat_id')

        if not user_input or not mbti_result or not user_id or not chat_id:
            print("❌ Data kurang:", user_input, mbti_result, user_id, chat_id)
            return jsonify({'error': 'user_input, mbti_result, user_id, dan chat_id wajib diisi'}), 400
        
        if user_id not in chat_histories:
            chat_histories[user_id] = {}
        if chat_id not in chat_histories[user_id]:
            chat_histories[user_id][chat_id] = []

        emotion, instruction, response, responseTime = chatbot_response(
            user_input, chat_histories[user_id][chat_id], mbti_result
        )
        chat_histories[user_id][chat_id].append((user_input, response))

        try:
            save_chat_response = requests.post(
                "http://localhost:3000/save-chat",
                json={
                    "userId": user_id,
                    "chatId": chat_id,
                    "text": user_input,
                    "response": response,
                    "emotion": emotion,
                    "responseTime": responseTime
                }
            )
            if save_chat_response.status_code != 200:
                print("❗ Gagal menyimpan chat ke backend Express:", save_chat_response.text)
        except Exception as save_error:
            print("❗ Error menyimpan chat ke backend Express:", str(save_error))

        return jsonify({
            'emotion': emotion,
            'instruction': instruction,
            'response': response,
            'responseTime': responseTime
        }), 200

    except Exception as e:
        print("❌ ERROR SERVER:", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/user', methods=['GET'])
def get_user():
    try:
        auth_header = request.headers.get('Authorization')
        user_id = request.args.get('id')

        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Token tidak valid'}), 401
        if not user_id:
            return jsonify({'error': 'ID user wajib disertakan sebagai query parameter'}), 400

        token = auth_header.split(' ')[1]
        headers = {"Authorization": f"Bearer {token}"}

        res = requests.get("http://localhost:3000/users", headers=headers)
        if res.status_code != 200:
            return jsonify({'error': 'Gagal mengambil data pengguna'}), res.status_code

        data = res.json()
        users = data.get('users', [])
        user = next((u for u in users if str(u.get('id')) == str(user_id)), None)

        if not user:
            return jsonify({'error': 'User tidak ditemukan'}), 404

        return jsonify({
            'id': user.get('id'),
            'name': user.get('name'),
            'email': user.get('email'),
            'mbtiResult': user.get('mbtiResult')
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summary', methods=['POST'])
def get_summary():
    try:
        data = request.get_json()
        print("Data received:", data)

        user_id = data.get('user_id')
        chat_id = data.get('chat_id')

        if not user_id or not chat_id:
            return jsonify({'error': 'user_id dan chat_id wajib diisi'}), 400

        if user_id not in chat_histories or chat_id not in chat_histories[user_id]:
            return jsonify({'summary': 'Tidak ada histori chat untuk sesi ini'}), 200

        chat_history = chat_histories[user_id][chat_id]
        if not chat_history:
            return jsonify({'summary': 'Histori chat kosong'}), 200

        summary = summarize_chat(chat_history)

        try:
            save_summary_response = requests.post(
                "http://localhost:3000/save-summary",
                json={
                    "userId": user_id,
                    "chatId": chat_id,
                    "summary": summary
                }
            )
            if save_summary_response.status_code != 200:
                print("❗ Gagal menyimpan summary ke backend Express:", save_summary_response.text)
        except Exception as save_summary_error:
            print("❗ Error menyimpan summary ke backend Express:", str(save_summary_error))

        return jsonify({'summary': summary}), 200

    except Exception as e:
        print("❌ ERROR SERVER:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)