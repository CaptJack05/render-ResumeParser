from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import os
import json
from werkzeug.utils import secure_filename
import re
from datetime import datetime
import PyPDF2
import docx2txt
import spacy
from collections import Counter
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Load .env for local dev ──────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)

# ── Config from environment (NEVER hardcoded) ────────────────────────────────
GOOGLE_AI_API_KEY = os.environ.get('GOOGLE_AI_API_KEY')
app.secret_key    = os.environ.get('SECRET_KEY', 'local-dev-secret-change-in-prod')

if not GOOGLE_AI_API_KEY:
    raise RuntimeError(
        "GOOGLE_AI_API_KEY is not set.\n"
        "  Local: add it to your .env file\n"
        "  Render: add it in Dashboard → Environment"
    )

# ── Database: PostgreSQL on Render, SQLite locally ───────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    if DATABASE_URL.startswith('postgres://'):          # Render gives old format
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    engine  = create_engine(DATABASE_URL)
    DB_TYPE = 'postgresql'
else:
    engine  = create_engine('sqlite:///resumes.db', connect_args={"check_same_thread": False})
    DB_TYPE = 'sqlite'

# ── Upload config ────────────────────────────────────────────────────────────
UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Google AI ────────────────────────────────────────────────────────────────
import google.generativeai as genai

try:
    genai.configure(api_key=GOOGLE_AI_API_KEY)
    generation_config = {
        "temperature":    0.1,
        "top_p":          0.95,
        "top_k":          40,
        "max_output_tokens": 8192,
    }
    model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)
    logger.info("Google AI initialised with Gemini 2.5 Flash")
except Exception as e:
    logger.error(f"Failed to initialise Google AI: {e}")
    model = None

# ── spaCy ────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

# ── Template filter ──────────────────────────────────────────────────────────
@app.template_filter('from_json')
def from_json_filter(value):
    if value:
        try:
            return json.loads(value)
        except Exception:
            return []
    return []

# ── Database init ────────────────────────────────────────────────────────────
def init_database():
    serial = 'SERIAL' if DB_TYPE == 'postgresql' else 'INTEGER'
    pk     = 'PRIMARY KEY' if DB_TYPE == 'postgresql' else 'PRIMARY KEY AUTOINCREMENT'

    ddl = f'''
        CREATE TABLE IF NOT EXISTS resumes (
            id                  {serial} {pk},
            filename            TEXT NOT NULL,
            name                TEXT,
            email               TEXT,
            phone               TEXT,
            skills              TEXT,
            current_location    TEXT,
            hometown            TEXT,
            education           TEXT,
            companies           TEXT,
            work_experience     TEXT,
            years_of_experience INTEGER,
            avg_work_duration   TEXT,
            certifications      TEXT,
            languages           TEXT,
            projects            TEXT,
            summary             TEXT,
            raw_text            TEXT,
            upload_date         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    '''
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()

# ── Helpers ──────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() or '' for p in reader.pages)
    except Exception as e:
        logger.error(f"PDF error: {e}"); return ""

def extract_text_from_docx(path):
    try:    return docx2txt.process(path)
    except Exception as e:
        logger.error(f"DOCX error: {e}"); return ""

def extract_text_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except Exception as e:
        logger.error(f"TXT error: {e}"); return ""

# ── AI Extraction ────────────────────────────────────────────────────────────
def extract_with_ai(text):
    if not model:
        return extract_with_basic_methods(text)

    prompt = f"""
You are an expert resume parser. Analyze the following resume text and extract ALL information into a well-structured JSON format.

IMPORTANT INSTRUCTIONS:
1. Return ONLY valid JSON — no markdown, no extra text, no code fences
2. Extract every piece of information present
3. Use null for missing fields, [] for missing arrays
4. Infer total years of experience from dates if not explicit
5. Calculate average work duration from job date ranges

REQUIRED JSON STRUCTURE:
{{
  "name": "Full name",
  "email": "email@example.com",
  "phone": "phone as string",
  "skills": ["skill1", "skill2"],
  "current_location": "City, Country",
  "hometown": "hometown if mentioned",
  "education": [
    {{"degree":"B.Tech","institution":"IIT Delhi","year":"2020","field":"Computer Science"}}
  ],
  "companies": ["Company A", "Company B"],
  "work_experience": [
    {{"company":"Acme","position":"SWE","duration":"Jan 2021 - Dec 2023","description":"Built APIs"}}
  ],
  "years_of_experience": 4,
  "avg_work_duration": "1.5 years",
  "certifications": ["AWS Certified", "Google Cloud"],
  "languages": ["English", "Hindi"],
  "projects": [
    {{"name":"My App","description":"What it does","technologies":["Python","React"]}}
  ],
  "summary": "2-3 sentence professional summary"
}}

RESUME TEXT:
{text[:6000]}

Return JSON only:
"""
    try:
        response = model.generate_content(prompt)
        if response and response.text:
            jt = response.text.strip()
            if '```json' in jt: jt = jt.split('```json')[1].split('```')[0]
            elif '```' in jt:   jt = jt.split('```')[1].split('```')[0]
            return validate_extracted_data(json.loads(jt.strip()))
    except json.JSONDecodeError:
        return _retry_extract(text)
    except Exception as e:
        logger.error(f"AI extraction error: {e}")
    return extract_with_basic_methods(text)

def _retry_extract(text):
    if not model: return extract_with_basic_methods(text)
    try:
        prompt = f"""Parse resume to JSON only (no markdown):
{{"name":"","email":"","phone":"","skills":[],"education":[],"companies":[],"years_of_experience":0,"summary":""}}
Resume: {text[:4000]}"""
        r = model.generate_content(prompt)
        if r and r.text:
            jt = re.sub(r'^```json\s*|\s*```$', '', r.text.strip())
            return validate_extracted_data(json.loads(jt))
    except Exception: pass
    return extract_with_basic_methods(text)

def validate_extracted_data(d):
    return {
        'name':                clean_str(d.get('name')),
        'email':               valid_email(d.get('email')),
        'phone':               valid_phone(d.get('phone')),
        'skills':              to_list(d.get('skills')),
        'current_location':    clean_str(d.get('current_location')),
        'hometown':            clean_str(d.get('hometown')),
        'education':           to_list(d.get('education')),
        'companies':           to_list(d.get('companies')),
        'work_experience':     to_list(d.get('work_experience')),
        'years_of_experience': valid_years(d.get('years_of_experience')),
        'avg_work_duration':   clean_str(d.get('avg_work_duration')),
        'certifications':      to_list(d.get('certifications')),
        'languages':           to_list(d.get('languages')),
        'projects':            to_list(d.get('projects')),
        'summary':             clean_str(d.get('summary')),
    }

def clean_str(v):
    if not v or v in ['null','None','N/A']: return None
    return str(v).strip()

def to_list(v):
    if v is None or v == 'null': return []
    if isinstance(v, list):      return [x for x in v if x and x != 'null']
    if isinstance(v, str) and v.strip() and v != 'null': return [v]
    return []

def valid_email(e):
    if not e or e in ['null','None']: return None
    if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', str(e)):
        return str(e).lower()
    return None

def valid_phone(p):
    if not p or p in ['null','None']: return None
    c = re.sub(r'[^\d+]', '', str(p))
    return c if len(c) >= 10 else None

def valid_years(y):
    if y is None or y in ['null','None']: return None
    try:
        if isinstance(y, str):
            m = re.search(r'\d+', y)
            y = m.group() if m else None
        n = int(y)
        return n if 0 <= n <= 50 else None
    except (ValueError, TypeError): return None

# ── Basic fallback extraction ─────────────────────────────────────────────────
def extract_with_basic_methods(text):
    email = _email(text)
    return {
        'name':                _name_spacy(text, email),
        'email':               email,
        'phone':               _phone(text),
        'skills':              _skills(text),
        'current_location':    _location(text),
        'hometown':            None,
        'education':           [_education(text)] if _education(text) else [],
        'companies':           _companies(text),
        'work_experience':     [],
        'years_of_experience': _years_exp(text),
        'avg_work_duration':   None,
        'certifications':      _certs(text),
        'languages':           [],
        'projects':            [],
        'summary':             _summary(text),
    }

def _email(text):
    m = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    return m[0] if m else None

def _phone(text):
    for pat in [r'\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', r'\b\d{10}\b']:
        m = re.findall(pat, text)
        if m: return m[0]
    return None

def _location(text):
    for pat in [r'(?:Location|Address|City):\s*([A-Z][a-zA-Z\s,]+)',
                r'(?:Based in|Living in|Residing in)\s*([A-Z][a-zA-Z\s,]+)']:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None

def _skills(text):
    kw = ['Python','Java','JavaScript','TypeScript','C++','C#','Ruby','PHP','Swift','Kotlin',
          'HTML','CSS','React','Angular','Vue','Node.js','Express','Django','Flask','FastAPI',
          'Spring','SQL','MySQL','PostgreSQL','MongoDB','Redis','Docker','Kubernetes',
          'AWS','Azure','GCP','Git','Jenkins','CI/CD','Machine Learning','AI',
          'Deep Learning','TensorFlow','PyTorch','Pandas','NumPy','REST API',
          'GraphQL','Microservices','DevOps','Linux','Selenium','Tableau','Power BI']
    tl = text.lower()
    return [s for s in kw if s.lower() in tl]

def _companies(text):
    cos = set()
    for pat in [r'(?:at|@)\s+([A-Z][A-Za-z0-9\s&.,]+?)(?:\s+as\s+|\s+—\s+|\s+-\s+)',
                r'([A-Z][A-Za-z0-9\s&.,]+?)(?:\s*-\s*(?:Software|Developer|Engineer|Manager|Analyst))']:
        for m in re.findall(pat, text):
            c = m.strip()
            if 2 < len(c) < 50: cos.add(c)
    return list(cos)[:10]

def _education(text):
    lines = []
    for kw in [r'B\.?Tech',r'M\.?Tech',r'MBA',r'BCA',r'MCA',r'B\.?Sc',r'M\.?Sc',r'Bachelor',r'Master',r'PhD']:
        lines.extend(re.findall(rf'{kw}[^.]*(?:\.|$)', text, re.IGNORECASE))
    return ' | '.join(lines[:3]) if lines else None

def _certs(text):
    certs = []
    for line in text.split('\n'):
        if any(k in line.lower() for k in ['certified','certification','certificate']):
            certs.append(line.strip())
    return certs[:5]

def _years_exp(text):
    for pat in [r'(\d+)\+?\s*years?\s+(?:of\s+)?experience', r'experience[:\s]+(\d+)\+?\s*years?']:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return int(m.group(1))
    return None

def _summary(text):
    for line in text.split('\n'):
        l = line.strip()
        if 50 < len(l) < 300 and any(w in l.lower() for w in ['experience','professional','seeking','skilled']):
            return l
    return None

def _name_spacy(text, email=None):
    if not nlp: return _name_basic(text, email)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    ignore = ['curriculum vitae','resume','cv','profile','contact','student','engineer','developer']
    for line in lines[:10]:
        if line.lower().startswith("name:"):
            c = line.split(":", 1)[1].strip()
            if _valid_name(c): return c
    for line in lines[:10]:
        if any(w in line.lower() for w in ignore): continue
        doc = nlp(line)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and _valid_name(ent.text): return ent.text
    return _name_basic(text, email)

def _name_basic(text, email=None):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    for line in lines[:10]:
        if '@' in line or re.search(r'\d{3,}', line): continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            return line
    if email:
        username = re.sub(r'\d+', '', email.split('@')[0])
        parts = re.findall(r'[A-Z][a-z]*', username)
        if parts: return " ".join(p.capitalize() for p in parts)
    return None

def _valid_name(c):
    if not c: return False
    if any(x in c.lower() for x in ['http','www','.com','portfolio']): return False
    return 2 <= len(c.split()) <= 4

# ── Parse + Save ─────────────────────────────────────────────────────────────
def parse_resume(file_path, filename):
    ext = filename.lower().rsplit('.', 1)[-1]
    text = {'pdf': extract_text_from_pdf, 'docx': extract_text_from_docx,
            'txt': extract_text_from_txt}.get(ext, lambda _: "")( file_path )
    if not text or len(text.strip()) < 50:
        return None
    logger.info(f"Parsing {filename} ({len(text)} chars) via {DB_TYPE}")
    parsed = extract_with_ai(text)
    parsed['filename'] = filename
    parsed['raw_text']  = text[:2000]
    return parsed

def save_to_database(d):
    params = {
        'filename':            d['filename'],
        'name':                d['name'],
        'email':               d['email'],
        'phone':               d['phone'],
        'skills':              json.dumps(d['skills']),
        'current_location':    d['current_location'],
        'hometown':            d['hometown'],
        'education':           json.dumps(d['education']),
        'companies':           json.dumps(d['companies']),
        'work_experience':     json.dumps(d['work_experience']),
        'years_of_experience': d['years_of_experience'],
        'avg_work_duration':   d['avg_work_duration'],
        'certifications':      json.dumps(d['certifications']),
        'languages':           json.dumps(d['languages']),
        'projects':            json.dumps(d['projects']),
        'summary':             d['summary'],
        'raw_text':            d['raw_text'],
    }
    cols = ','.join(params.keys())
    vals = ','.join(f':{k}' for k in params.keys())

    with engine.connect() as conn:
        if DB_TYPE == 'postgresql':
            row = conn.execute(
                text(f'INSERT INTO resumes ({cols}) VALUES ({vals}) RETURNING id'), params
            ).fetchone()
            rid = row[0]
        else:
            result = conn.execute(text(f'INSERT INTO resumes ({cols}) VALUES ({vals})'), params)
            rid = result.lastrowid
        conn.commit()
    return rid

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    with engine.connect() as conn:
        total   = conn.execute(text("SELECT COUNT(*) FROM resumes")).fetchone()[0]
        recent  = conn.execute(text(
            "SELECT id,name,email,years_of_experience,upload_date FROM resumes ORDER BY upload_date DESC LIMIT 5"
        )).fetchall()
        skills_raw = conn.execute(text("SELECT skills FROM resumes WHERE skills IS NOT NULL")).fetchall()

    counter = Counter()
    for row in skills_raw:
        try: counter.update(json.loads(row[0]))
        except Exception: pass

    return render_template('index.html', total=total, recent=recent,
                           top_skills=counter.most_common(8))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error'); return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error'); return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                parsed = parse_resume(filepath, filename)
                if parsed and parsed.get('name'):
                    rid = save_to_database(parsed)
                    flash(f'"{filename}" parsed successfully!', 'success')
                    return redirect(url_for('view_resume', resume_id=rid))
                else:
                    flash('Could not extract enough information. Check the file.', 'error')
            except Exception as e:
                logger.error(f"Processing error: {e}")
                flash('Error parsing resume. Please try again.', 'error')
        else:
            flash('Invalid format. Upload PDF, DOCX, or TXT.', 'error')
    return render_template('upload.html')

@app.route('/search')
def search():
    q   = request.args.get('q', '')
    sk  = request.args.get('skill', '')
    exp = request.args.get('experience', '')
    results = []

    if q or sk or exp:
        sql, params = "SELECT * FROM resumes WHERE 1=1", {}
        if q:
            sql += " AND (name LIKE :q1 OR email LIKE :q2 OR companies LIKE :q3 OR summary LIKE :q4)"
            params.update(q1=f'%{q}%', q2=f'%{q}%', q3=f'%{q}%', q4=f'%{q}%')
        if sk:
            sql += " AND skills LIKE :sk"; params['sk'] = f'%{sk}%'
        if exp:
            try: sql += " AND years_of_experience >= :exp"; params['exp'] = int(exp)
            except ValueError: pass
        sql += " ORDER BY upload_date DESC"
        with engine.connect() as conn:
            results = conn.execute(text(sql), params).fetchall()

    return render_template('search.html', results=results, query=q, skill_filter=sk, experience_filter=exp)

@app.route('/resume/<int:resume_id>')
def view_resume(resume_id):
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM resumes WHERE id = :id"), {'id': resume_id}).fetchone()
    if not row:
        flash('Resume not found', 'error'); return redirect(url_for('index'))

    resume = {
        'id': row[0], 'filename': row[1], 'name': row[2], 'email': row[3], 'phone': row[4],
        'skills':              json.loads(row[5])  if row[5]  else [],
        'current_location':    row[6],
        'hometown':            row[7],
        'education':           json.loads(row[8])  if row[8]  else [],
        'companies':           json.loads(row[9])  if row[9]  else [],
        'work_experience':     json.loads(row[10]) if row[10] else [],
        'years_of_experience': row[11],
        'avg_work_duration':   row[12],
        'certifications':      json.loads(row[13]) if row[13] else [],
        'languages':           json.loads(row[14]) if row[14] else [],
        'projects':            json.loads(row[15]) if row[15] else [],
        'summary':             row[16],
        'upload_date':         row[18],
    }
    return render_template('view_resume.html', resume=resume)

@app.route('/api/stats')
def api_stats():
    with engine.connect() as conn:
        total   = conn.execute(text("SELECT COUNT(*) FROM resumes")).fetchone()[0]
        skills  = conn.execute(text("SELECT skills FROM resumes WHERE skills IS NOT NULL")).fetchall()
        avg_exp = conn.execute(text("SELECT AVG(years_of_experience) FROM resumes WHERE years_of_experience IS NOT NULL")).fetchone()[0]
    c = Counter()
    for row in skills:
        try: c.update(json.loads(row[0]))
        except Exception: pass
    return jsonify({'total_resumes': total, 'top_skills': dict(c.most_common(10)),
                    'average_experience': round(avg_exp, 1) if avg_exp else 0})

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    if len(data['text'].strip()) < 50:
        return jsonify({'error': 'Text too short'}), 400
    try:
        return jsonify({'success': True, 'data': extract_with_ai(data['text'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/resumes')
def api_list_resumes():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT id,filename,name,email,phone,years_of_experience,upload_date FROM resumes ORDER BY upload_date DESC"
        )).fetchall()
    return jsonify([{'id':r[0],'filename':r[1],'name':r[2],'email':r[3],
                     'phone':r[4],'years_of_experience':r[5],'upload_date':str(r[6])} for r in rows])

@app.route('/delete/<int:resume_id>', methods=['POST'])
def delete_resume(resume_id):
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM resumes WHERE id = :id"), {'id': resume_id})
        conn.commit()
    flash('Resume deleted.', 'success')
    return redirect(url_for('index'))

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_database()
    logger.info(f"AI Resume Parser started | DB: {DB_TYPE}")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
