from flask import Flask, request, redirect, url_for, render_template,send_from_directory

import os
import openai
import base64
import json
import sqlite3
import re
from dotenv import load_dotenv
import hashlib
from PIL import Image, ImageOps
import hashlib
import numpy as np
import faiss
import requests


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

THUMBNAIL_FOLDER = 'static/thumbnails'
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

# Initialize the database
DB_PATH = 'photo_metadata.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                description TEXT,
                emotion TEXT,
                action TEXT,
                keywords TEXT
            );
        ''')
        conn.commit()
    print(" * Database initialized")

# Initialize GPT
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_dim = 1536  # Output dimension of text-embedding-3-small
faiss_index = faiss.IndexFlatL2(embedding_dim)
id_mapping = []  # Used to record the mapping between photo.id in the database and the position of the faiss vector

def generate_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text
    )
    return response["data"][0]["embedding"]

def analyze_image(image_path):
    with open(image_path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")

    prompt = (
        "You are an expert visual analyst. "
        "Analyze the uploaded image and return the result strictly in JSON format like this:\n"
        "{ \"description\": \"...\", \"keywords\": [\"...\"], \"emotion\": \"...\", \"action\": \"...\" }\n\n"
        "- The `emotion` should reflect the overall emotional tone of the image.\n"
        "Output ONLY valid raw JSON. Do not wrap with ```json or commentary."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                }
            ]}
        ],
        max_tokens=800
    )

    raw_content = response["choices"][0]["message"]["content"]

    # Automatically strip Markdown wrapped with ```json
    match = re.search(r"```json\\s*(.*?)\\s*```", raw_content, re.DOTALL)
    if match:
        cleaned_json = match.group(1)
    else:
        cleaned_json = raw_content

    try:
        result = json.loads(cleaned_json)
        description = result.get("description", "")
        keywords = result.get("keywords", [])
        emotion = result.get("emotion", "")
        action = result.get("action", "")
    except Exception as e:
        description = raw_content
        keywords = []
        emotion = "Judged comprehensively by GPT"
        action = "Judged comprehensively by GPT"

    return description, keywords, emotion, action

def load_faiss_index():
    global faiss_index, id_mapping
    faiss_index.reset()
    id_mapping.clear()

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, embedding FROM photos WHERE embedding IS NOT NULL").fetchall()
        vectors = []
        for row in rows:
            photo_id, embedding_blob = row
            vec = np.frombuffer(embedding_blob, dtype=np.float32)
            if vec.shape[0] == embedding_dim:
                vectors.append(vec)
                id_mapping.append(photo_id)

    if vectors:
        faiss_index.add(np.stack(vectors))
        print(f"✅ Loaded {len(vectors)} vectors into FAISS")
    else:
        print("⚠️ No embeddings found to load")


@app.route('/manifest.json')
def manifest():
    return send_from_directory('static', 'manifest.json')


@app.route('/')
def index():
    query = request.args.get('q', '').strip().lower()
    safe_query = f'"{query}"'
    photos = []
    source_label = ""

    if query:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row

            # 文本搜索
            sql = """
                SELECT * FROM photos
                WHERE id IN (SELECT rowid FROM photos_fts WHERE photos_fts MATCH ?)
                   OR LOWER(emotion) LIKE ?
                   OR LOWER(action) LIKE ?
                   OR LOWER(keywords) LIKE ?
                ORDER BY id DESC LIMIT 10
            """
            params = [safe_query] + [f"%{query}%"] * 3
            photos = conn.execute(sql, params).fetchall()

            if photos:
                source_label = "Matched by keywords"
            else:
                # 语义搜索 fallback
                query_embedding = generate_embedding(query)
                D, I = faiss_index.search(np.array([query_embedding], dtype=np.float32), k=20)

                MAX_DISTANCE = 1.40  # 可调节范围：0.8 ~ 1.5
                results = []
                seen = set()

                for dist, idx in zip(D[0], I[0]):
                    if idx >= 0 and dist <= MAX_DISTANCE:
                        photo_id = id_mapping[idx]
                        if photo_id not in seen:
                            score = 1 / (1 + dist)  # 转换为相似度分数（越大越相似）
                            results.append((photo_id, score))
                            seen.add(photo_id)

                # 相似度从高到低排序
                results.sort(key=lambda x: x[1], reverse=True)
                matched_ids = [photo_id for photo_id, _ in results]

                if matched_ids:
                    placeholders = ",".join("?" * len(matched_ids))
                    sql = f"SELECT * FROM photos WHERE id IN ({placeholders})"
                    row_list = conn.execute(sql, matched_ids).fetchall()
                    # 按相似度顺序重新排序结果
                    photo_dict = {row["id"]: row for row in row_list}
                    photos = [photo_dict[pid] for pid in matched_ids if pid in photo_dict]
                    source_label = "Matched by semantic meaning"
    else:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            photos = conn.execute("SELECT * FROM photos ORDER BY id DESC LIMIT 10").fetchall()
            source_label = "Recently uploaded"

    return render_template('index.html', photos=photos, query=query, source_label=source_label)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = file.filename
    file_bytes = file.read()

    # First calculate the hash value (based on memory)
    file_hash = hashlib.md5(file_bytes).hexdigest()

    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = file_hash + ext

    # Check if there is the same image in the database
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        existing = conn.execute("SELECT id FROM photos WHERE file_hash = ?", (file_hash,)).fetchone()

    if existing:
        return render_template('upload_result.html',
                               filename=filename,
                               description="Upload failed: Duplicate image (same content).",
                               keywords="—",
                               emotion="—",
                               action="—")

    # Save the file (only write to disk for the first occurrence of the image)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(file_path, 'wb') as f:
        f.write(file_bytes)  # After saving the original image

    # Save the thumbnail
    thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)
    create_thumbnail(file_path, thumb_path)

    # Analyze the image and save the record
    description, keywords, emotion, action = analyze_image(file_path)
    # Write to the main table + FTS
    save_photo_metadata(filename, description, keywords, emotion, action, file_hash)

    load_faiss_index()

    return render_template('upload_result.html',
                           filename=filename,
                           description=description,
                           keywords=", ".join(keywords),
                           emotion=emotion,
                           action=action)


@app.route('/generate_music', methods=['POST'])
def generate_music():
    print("Generating music...")
    filename = request.form.get('filename')
    if not filename:
        return "Missing filename"

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM photos WHERE filename = ?", (filename,)).fetchone()

    if not row:
        return "Image metadata not found"

    prompt = f"A {row['emotion']} instrumental track inspired by: {row['keywords']}. Scene: {row['description']}"

    # 调用音乐生成 API
    music_data = call_stable_audio(prompt) 

    # 保存音乐文件
    music_filename = os.path.splitext(filename)[0]

    audio_path = f"static/audio/{music_filename}.mp3"
    os.makedirs("static/audio", exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(music_data)

    return render_template("music_result.html", image_filename=filename, filename=music_filename, audio_path=audio_path, prompt=prompt)


def calculate_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_photo_metadata(filename, description, keywords, emotion, action, file_hash):
    # 构建文本用于 embedding
    text_for_embedding = f"{description}. Emotion: {emotion}. Action: {action}. Keywords: {', '.join(keywords)}"
    embedding = generate_embedding(text_for_embedding)
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO photos (filename, description, emotion, action, keywords, file_hash, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (filename, description, emotion, action, ",".join(keywords), file_hash, embedding_bytes)
        )
        photo_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO photos_fts(rowid, filename, description, keywords) VALUES (?, ?, ?, ?)",
            (photo_id, filename, description, ",".join(keywords))
        )
        conn.commit()


def call_stable_audio(prompt):
    url = "https://api.stability.ai/v2beta/audio/stable-audio-2/text-to-audio"
    headers = {
        "Authorization": f"Bearer {os.getenv('STABLE_AUDIO_API_KEY')}",
        "Accept": "audio/*"  # Directly get the audio file
    }

    files = {
        "prompt": (None, prompt),
        "duration": (None, "10"),
        "output_format": (None, "mp3"),
        "model": (None, "stable-audio-2.0")
    }

    response = requests.post(url, headers=headers, files=files)
    if response.status_code != 200:
        raise Exception(f"Stable Audio failed: {response.status_code} - {response.text}")
    
    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Size:", len(response.content))

    return response.content  # MP3 audio file


def create_thumbnail(input_path, output_path, size=(300, 300)):
    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)  # Automatically rotate according to EXIF
        img.thumbnail(size)
        img.save(output_path)

def debug_faiss_search(query):
    import sqlite3
    import numpy as np

    # Query all embeddings and descriptions
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, filename, description, emotion, action, keywords, embedding FROM photos").fetchall()

    print("\n📸 Current database content:")
    id_map = {}
    for row in rows:
        pid = row[0]
        filename = row[1]
        desc = row[2]
        keywords = row[5]
        emb = row[6]
        print(f"  ID={pid} | file={filename} | desc={desc[:30]} | keywords={keywords} | emb_size={len(emb) if emb else 0}")
        if emb:
            vec = np.frombuffer(emb, dtype=np.float32)
            id_map[pid] = vec

    if not id_map:
        print("❌ No valid embedding, possibly failed to save")
        return

    # Query semantic vector
    query_vec = generate_embedding(query)
    D, I = faiss_index.search(np.array([query_vec], dtype=np.float32), k=5)

    print("\n🔍 Search term =", query)
    print("  FAISS returned index =", I[0])
    print("  Corresponding photo IDs =", [id_mapping[i] for i in I[0]])

    print("\n🧠 Matched photo information:")
    for i in I[0]:
        photo_id = id_mapping[i]
        row = next(r for r in rows if r[0] == photo_id)
        print(f"  [{photo_id}] desc: {row[2][:50]} | keywords: {row[5]} | emotion: {row[3]} | action: {row[4]}")


if __name__ == '__main__':
    init_db()
    load_faiss_index()
    app.run(host='0.0.0.0', port=8080, debug=True)
