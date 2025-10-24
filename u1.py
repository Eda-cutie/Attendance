# app.py
import streamlit as st
from streamlit_javascript import st_javascript
import qrcode
from PIL import Image
import io, base64, os, sqlite3, hashlib, datetime, math, random, pandas as pd
from dateutil import parser
import plotly.express as px
import numpy as np
from pyzbar.pyzbar import decode as qr_decode  # may require zbar system lib; fallback exists

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Attendance & Analytics", layout="wide", initial_sidebar_state="expanded")
DB_PATH = "attendance.db"
QR_EXPIRE_SECONDS = 300  # 5 minutes
GEOFENCE_RADIUS_METERS = 50  # default acceptable radius

# ---------------------------
# Utilities
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # users: id, email, name, role, password_hash
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 email TEXT UNIQUE,
                 name TEXT,
                 role TEXT,
                 password_hash TEXT
                 )''')
    # courses
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 code TEXT,
                 title TEXT,
                 department TEXT
                 )''')
    # lectures: id, course_id, lecture_dt, qr_token, qr_created_at, lat, lon, radius
    c.execute('''CREATE TABLE IF NOT EXISTS lectures (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 course_id INTEGER,
                 lecture_dt TEXT,
                 qr_token TEXT,
                 qr_created_at TEXT,
                 lat REAL,
                 lon REAL,
                 radius INTEGER,
                 FOREIGN KEY(course_id) REFERENCES courses(id)
                 )''')
    # attendance: id, user_id, lecture_id, timestamp, lat, lon, status
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 lecture_id INTEGER,
                 timestamp TEXT,
                 lat REAL,
                 lon REAL,
                 status TEXT,
                 qr_token TEXT,
                 UNIQUE(user_id, lecture_id),
                 FOREIGN KEY(user_id) REFERENCES users(id),
                 FOREIGN KEY(lecture_id) REFERENCES lectures(id)
                 )''')
    # puzzle tracking & badges
    c.execute('''CREATE TABLE IF NOT EXISTS puzzles_completed (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 puzzle_date TEXT,
                 puzzle_type TEXT,
                 time_seconds INTEGER,
                 FOREIGN KEY(user_id) REFERENCES users(id)
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS badges (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 badge TEXT,
                 awarded_at TEXT,
                 FOREIGN KEY(user_id) REFERENCES users(id)
                 )''')
    conn.commit()
    return conn

conn = init_db()
c = conn.cursor()

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def seed_demo_data():
    # Add an admin and a faculty and sample students if not exist
    users = c.execute("SELECT count(*) FROM users").fetchone()[0]
    if users == 0:
        c.execute("INSERT OR IGNORE INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)",
                  ("admin@college.edu", "Admin User", "admin", hash_password("admin123")))
        c.execute("INSERT OR IGNORE INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)",
                  ("prof@college.edu", "Prof Alice", "faculty", hash_password("prof123")))
        for i in range(1, 8):
            c.execute("INSERT OR IGNORE INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)",
                      (f"student{i}@college.edu", f"Student {i}", "student", hash_password("student123")))
        conn.commit()
seed_demo_data()

def distance_meters(lat1, lon1, lat2, lon2):
    # Haversine formula
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def make_qr_image(token):
    qr = qrcode.QRCode(box_size=6, border=2)
    qr.add_data(token)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def generate_token():
    return hashlib.sha1(os.urandom(64)).hexdigest()

def now_iso():
    return datetime.datetime.utcnow().isoformat()

# ---------------------------
# Authentication (simple)
# ---------------------------
def login_form():
    st.sidebar.title("Login / Register")
    mode = st.sidebar.radio("Mode", ["Login", "Register"])
    email = st.sidebar.text_input("Email")
    pwd = st.sidebar.text_input("Password", type="password")
    if mode == "Register":
        name = st.sidebar.text_input("Full name")
        role = st.sidebar.selectbox("Role", ["student", "faculty"])
        if st.sidebar.button("Register"):
            try:
                c.execute("INSERT INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)",
                          (email, name, role, hash_password(pwd)))
                conn.commit()
                st.sidebar.success("Registered — please login.")
            except sqlite3.IntegrityError:
                st.sidebar.error("Email already registered.")
        st.sidebar.info("Admin account pre-seeded: admin@college.edu / admin123")
        return None
    else:
        if st.sidebar.button("Login"):
            row = c.execute("SELECT id, name, role, password_hash FROM users WHERE email=?", (email,)).fetchone()
            if not row:
                st.sidebar.error("User not found.")
                return None
            uid, name, role, phash = row
            if hash_password(pwd) == phash:
                st.session_state['user'] = {"id": uid, "email": email, "name": name, "role": role}
                st.sidebar.success(f"Logged in as {name} ({role})")
                return st.session_state['user']
            else:
                st.sidebar.error("Incorrect password.")
                return None
    return None

def logout():
    if 'user' in st.session_state:
        del st.session_state['user']

# Set up user session
user = st.session_state.get('user', None)
if not user:
    u = login_form()
    if u:
        user = u

# ---------------------------
# Pages
# ---------------------------
def sidebar_logout():
    if user:
        st.sidebar.markdown(f"**Signed in:** {user['name']} ({user['role']})")
        if st.sidebar.button("Logout"):
            logout()
            st.experimental_rerun()

sidebar_logout()

st.title("📚 Attendance & Analytics — QR + GPS (Streamlit MVP)")
st.markdown("A demo system with role-based access, QR check-in, GPS validation, analytics, and daily puzzles.")

# Helper: role gate
def require_role(roles):
    if not user:
        st.warning("Please login via left sidebar.")
        st.stop()
    if user['role'] not in roles:
        st.error("Access denied for your role.")
        st.stop()

# ---------------------------
# Admin Tools
# ---------------------------
def admin_page():
    require_role(['admin'])
    st.header("Admin Console — Manage Users, Courses, Lectures, and Analytics")
    tabs = st.tabs(["Users", "Courses & Lectures", "Analytics (Editable)", "QR Management", "Notifications"])
    # Users
    with tabs[0]:
        st.subheader("Users")
        if st.button("Refresh Users"):
            pass
        users_df = pd.read_sql_query("SELECT id, email, name, role FROM users", conn)
        st.dataframe(users_df, use_container_width=True)
        with st.expander("Add User"):
            email = st.text_input("Email", key="uemail")
            name = st.text_input("Name", key="uname")
            role = st.selectbox("Role", ["student", "faculty", "admin"], key="urole")
            pwd = st.text_input("Password", key="upwd")
            if st.button("Create User", key="create_user"):
                try:
                    c.execute("INSERT INTO users (email, name, role, password_hash) VALUES (?, ?, ?, ?)",
                              (email, name, role, hash_password(pwd)))
                    conn.commit()
                    st.success("User created")
                except sqlite3.IntegrityError:
                    st.error("Email exists")
    # Courses & Lectures
    with tabs[1]:
        st.subheader("Courses & Lectures")
        if st.button("Refresh Courses"):
            pass
        courses_df = pd.read_sql_query("SELECT * FROM courses", conn)
        st.dataframe(courses_df)
        with st.form("add_course"):
            code = st.text_input("Course Code")
            title = st.text_input("Title")
            dept = st.text_input("Department")
            submitted = st.form_submit_button("Add Course")
            if submitted:
                c.execute("INSERT INTO courses (code, title, department) VALUES (?, ?, ?)", (code, title, dept))
                conn.commit()
                st.success("Course added")
        st.markdown("---")
        st.subheader("Create Lecture / QR")
        courses = c.execute("SELECT id, code || ' - ' || title FROM courses").fetchall()
        course_map = {str(r[0]): r[1] for r in courses}
        if courses:
            course_choice = st.selectbox("Course", options=list(course_map.keys()), format_func=lambda x: course_map[x])
            dt = st.datetime_input("Lecture Date & Time", value=datetime.datetime.utcnow())
            lat = st.number_input("Latitude (classroom)", value=0.0, format="%.6f")
            lon = st.number_input("Longitude (classroom)", value=0.0, format="%.6f")
            rad = st.number_input("Geofence Radius (meters)", value=GEOFENCE_RADIUS_METERS)
            if st.button("Generate Lecture QR"):
                token = generate_token()
                c.execute("INSERT INTO lectures (course_id, lecture_dt, qr_token, qr_created_at, lat, lon, radius) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (int(course_choice), dt.isoformat(), token, now_iso(), lat, lon, int(rad)))
                conn.commit()
                st.success("Lecture created; QR token generated")
        else:
            st.info("Add at least one course first.")
    # Analytics editable
    with tabs[2]:
        st.subheader("Editable Analytics")
        st.info("Admin can edit attendance records here.")
        lectures = pd.read_sql_query("SELECT l.id, c.code || ' - ' || c.title as course, l.lecture_dt FROM lectures l JOIN courses c ON l.course_id=c.id", conn)
        if lectures.empty:
            st.write("No lectures yet.")
        else:
            st.dataframe(lectures)
            lid = st.number_input("Lecture ID to view/edit attendance", min_value=1, step=1)
            if st.button("Load Attendance"):
                att = pd.read_sql_query("SELECT a.id, u.name AS student, a.timestamp, a.status FROM attendance a JOIN users u ON a.user_id = u.id WHERE lecture_id=?",
                                        conn, params=(lid,))
                st.dataframe(att)
                if not att.empty:
                    sid = st.number_input("Attendance ID to edit", min_value=1, step=1)
                    new_status = st.selectbox("New Status", ["present", "absent", "late"])
                    if st.button("Update Status"):
                        c.execute("UPDATE attendance SET status=? WHERE id=?", (new_status, sid))
                        conn.commit()
                        st.success("Updated")
    with tabs[3]:
        st.subheader("QR Management")
        qrs = pd.read_sql_query("SELECT id, qr_token, qr_created_at FROM lectures", conn)
        st.dataframe(qrs)
        if st.button("Show Latest QR Images"):
            latest = c.execute("SELECT id, qr_token, qr_created_at FROM lectures ORDER BY id DESC LIMIT 6").fetchall()
            cols = st.columns(3)
            for i, row in enumerate(latest):
                img = make_qr_image(row[1])
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                cols[i%3].image(buf.getvalue(), caption=f"Lecture {row[0]} token (created {row[2]})")
    with tabs[4]:
        st.subheader("Notifications")
        st.write("Send email notifications to students (sample demo using SMTP config).")
        # demo only; production would integrate with Twilio/Firebase
        to_email = st.text_input("To email")
        subj = st.text_input("Subject")
        body = st.text_area("Message")
        if st.button("Send (demo)"):
            st.info("Demo: configured SMTP not included. In production, integrate with SMTP/Twilio/FCM.")
    st.markdown("**Admin tips:** Use the editable analytics to correct any attendance errors for your demo.")

# ---------------------------
# Faculty Page
# ---------------------------
def faculty_page():
    require_role(['faculty', 'admin'])
    st.header("Faculty Dashboard (Read-Only Analytics)")
    st.markdown("Generate QR codes for your lectures and view attendance analytics (read-only for faculty).")
    # Generate QR (faculty can generate too if admin allowed)
    courses = c.execute("SELECT id, code || ' - ' || title FROM courses").fetchall()
    course_map = {str(r[0]): r[1] for r in courses}
    if courses:
        course_choice = st.selectbox("Choose Course", options=list(course_map.keys()), format_func=lambda x: course_map[x])
        dt = st.datetime_input("Lecture Date & Time", value=datetime.datetime.utcnow())
        lat = st.number_input("Latitude (classroom)", value=0.0, format="%.6f")
        lon = st.number_input("Longitude (classroom)", value=0.0, format="%.6f")
        rad = st.number_input("Geofence Radius (meters)", value=GEOFENCE_RADIUS_METERS)
        if st.button("Generate Lecture QR (Faculty)"):
            token = generate_token()
            c.execute("INSERT INTO lectures (course_id, lecture_dt, qr_token, qr_created_at, lat, lon, radius) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (int(course_choice), dt.isoformat(), token, now_iso(), lat, lon, int(rad)))
            conn.commit()
            st.success("Lecture created; QR token generated")
            st.image(make_qr_image(token))
    else:
        st.info("No courses available. Admin create courses.")
    st.markdown("---")
    st.subheader("View Class Attendance (Read-Only)")
    lectures = pd.read_sql_query("SELECT l.id, c.code || ' - ' || c.title as course, l.lecture_dt FROM lectures l JOIN courses c ON l.course_id=c.id ORDER BY l.lecture_dt DESC", conn)
    st.dataframe(lectures)
    lid = st.number_input("Lecture ID to view", min_value=1, step=1)
    if st.button("Load Attendance for Lecture"):
        att = pd.read_sql_query("SELECT a.id, u.name AS student, a.timestamp, a.status FROM attendance a JOIN users u ON a.user_id = u.id WHERE lecture_id=?",
                                conn, params=(lid,))
        st.dataframe(att)
    # faculty analytics read-only
    st.markdown("---")
    st.subheader("Analytics (Read-Only)")
    agg = pd.read_sql_query("""SELECT u.name, c.code || ' - ' || c.title as course,
                              SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) as present_count,
                              COUNT(a.id) as total_sessions
                              FROM attendance a
                              JOIN users u ON a.user_id=u.id
                              JOIN lectures l ON a.lecture_id=l.id
                              JOIN courses c ON l.course_id=c.id
                              GROUP BY u.id, c.id LIMIT 200""", conn)
    if not agg.empty:
        agg['attendance_pct'] = (agg['present_count'] / agg['total_sessions'] * 100).round(1)
        fig = px.bar(agg.sort_values("attendance_pct", ascending=False).head(20), x="name", y="attendance_pct", color="course",
                     title="Top 20 student attendance % (sample)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No attendance data yet.")

# ---------------------------
# Student Page
# ---------------------------
def student_page():
    require_role(['student'])
    st.header("Student Home — Check In & Daily Mind Workout")
    st.markdown(f"Welcome, **{user['name']}**! Use the QR scanner to check into lectures and try today's puzzles.")

    # Show today's lectures (simple)
    today = datetime.datetime.utcnow().date()
    lectures_today = pd.read_sql_query("SELECT l.id, c.code || ' - ' || c.title as course, l.lecture_dt FROM lectures l JOIN courses c ON l.course_id=c.id WHERE DATE(lecture_dt)=?",
                                      conn, params=(today.isoformat(),))
    st.subheader("Today's Lectures")
    if lectures_today.empty:
        st.info("No lectures scheduled for today (UTC date).")
    else:
        st.dataframe(lectures_today)

    st.markdown("### QR Check-in")
    st.write("Either capture the QR with your camera or paste the QR token text. Location will be verified.")
    # Try to get browser geolocation via JS
    lat, lon = None, None
    js_available = True
    try:
        # This will request permission through browser and return coords or None
        js_code = """
        async function getCoords(){
            return new Promise((resolve, reject) => {
                if (!navigator.geolocation){
                    resolve({lat:null, lon:null});
                } else {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                        (err) => resolve({lat:null, lon:null})
                    );
                }
            });
        }
        getCoords();
        """
        coords = st_javascript(js_code, key="geoloc")
        if coords and isinstance(coords, dict) and coords.get('lat') is not None:
            lat = coords.get('lat'); lon = coords.get('lon')
            st.success(f"Location acquired: {lat:.6f}, {lon:.6f}")
        else:
            st.info("Browser location not available or permission denied. You can enter coords manually.")
            js_available = False
    except Exception:
        js_available = False

    col1, col2 = st.columns(2)
    with col1:
        img_file = st.camera_input("Scan QR (use camera) — capture image of QR")
        manual_qr = st.text_input("Or paste QR token text")
    with col2:
        if not js_available:
            lat = st.number_input("Latitude", value=0.0, format="%.6f")
            lon = st.number_input("Longitude", value=0.0, format="%.6f")
        lecture_id = st.number_input("Lecture ID to check in for", min_value=1, step=1)
        if st.button("Submit Check-In"):
            token = None
            if img_file:
                try:
                    img = Image.open(img_file)
                    decoded = qr_decode(img)
                    if decoded:
                        token = decoded[0].data.decode('utf-8')
                        st.success("QR decoded from image.")
                    else:
                        st.warning("Couldn't decode QR from image; please paste token.")
                except Exception as e:
                    st.error(f"QR decode error: {e}")
            if not token:
                token = manual_qr.strip() or None
            if not token:
                st.error("No QR token provided.")
            else:
                # validate lecture exists and token valid
                lec = c.execute("SELECT id, qr_token, qr_created_at, lat, lon, radius FROM lectures WHERE id=?", (lecture_id,)).fetchone()
                if not lec:
                    st.error("Lecture not found")
                else:
                    _, lec_token, lec_created_at, lec_lat, lec_lon, lec_rad = lec
                    # check token match & expiry
                    created_at = parser.isoparse(lec_created_at) if lec_created_at else None
                    if token != lec_token:
                        st.error("QR token does not match lecture.")
                    else:
                        # expiry check
                        if created_at:
                            age = (datetime.datetime.utcnow() - created_at).total_seconds()
                            if age > QR_EXPIRE_SECONDS:
                                st.error("QR has expired.")
                                return
                        # location check
                        if lat is None or lon is None or lec_lat is None or lec_lon is None:
                            st.error("Location data missing — cannot verify geofence.")
                        else:
                            dist = distance_meters(lat, lon, lec_lat, lec_lon)
                            status = "present" if dist <= lec_rad else "absent"
                            try:
                                c.execute("INSERT INTO attendance (user_id, lecture_id, timestamp, lat, lon, status, qr_token) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                          (user['id'], lecture_id, now_iso(), lat, lon, status, token))
                                conn.commit()
                                st.success(f"Check-in recorded. Status: {status} (distance {int(dist)} m)")
                                # award badge for on-time perfect? example
                                if status == "present":
                                    # check streaks or awarding simple badge:
                                    st.balloons()
                            except sqlite3.IntegrityError:
                                st.warning("You already checked in for this lecture.")
    st.markdown("---")
    # Student attendance view
    st.subheader("My Attendance Summary")
    df = pd.read_sql_query("SELECT l.lecture_dt, c.code || ' - ' || c.title as course, a.status FROM attendance a JOIN lectures l ON a.lecture_id=l.id JOIN courses c ON l.course_id=c.id WHERE a.user_id=? ORDER BY l.lecture_dt DESC", conn, params=(user['id'],))
    if df.empty:
        st.info("No attendance records yet.")
    else:
        st.dataframe(df)
        pct = (df['status']=='present').sum() / len(df) * 100
        st.metric("Overall attendance %", f"{pct:.1f}%")

    st.markdown("### Daily Mind Workout (Puzzles)")
    puzzle_tabs = st.tabs(["Connect the Dots", "Word Search", "Sudoku"])
    today_str = datetime.date.today().isoformat()
    # Connect-the-Dots (simple randomized points)
    with puzzle_tabs[0]:
        st.write("Connect the Dots — draw lines connecting numbered points (simulated).")
        # We'll provide static interactive: show numbered points and ask student to 'complete' as they see fit
        random.seed(today_str)
        pts = [(random.randint(10,90), random.randint(10,90)) for _ in range(8)]
        df_pts = pd.DataFrame(pts, columns=["x","y"])
        fig = px.scatter(df_pts, x="x", y="y", text=df_pts.index+1, title="Today's Connect-the-Dots")
        st.plotly_chart(fig)
        if st.button("Mark Connect-the-Dots Completed"):
            c.execute("INSERT INTO puzzles_completed (user_id, puzzle_date, puzzle_type, time_seconds) VALUES (?, ?, ?, ?)",
                      (user['id'], today_str, "connect", 60))
            conn.commit()
            st.success("Marked completed — good job! Badge awarded.")
            c.execute("INSERT INTO badges (user_id, badge, awarded_at) VALUES (?, ?, ?)",
                      (user['id'], "Dot Master", now_iso()))
            conn.commit()
    # Word Search
    with puzzle_tabs[1]:
        st.write("Word Search — find words in the grid.")
        # simple small generator using a fixed word list
        words = ["STREAM", "PYTHON", "DATA", "LOGIC", "CODE"]
        grid = [[" "]*12 for _ in range(12)]
        # place words horizontally for demo
        for i, w in enumerate(words):
            row = i*2
            for j,ch in enumerate(w):
                grid[row][j] = ch
        grid_text = "\n".join(" ".join(row) for row in grid)
        st.text_area("Wordsearch", grid_text, height=220)
        st.write("Find words:", ", ".join(words))
        if st.button("Mark Wordsearch Completed"):
            c.execute("INSERT INTO puzzles_completed (user_id, puzzle_date, puzzle_type, time_seconds) VALUES (?, ?, ?, ?)",
                      (user['id'], today_str, "wordsearch", 120))
            conn.commit()
            st.success("Completed! Points awarded.")
    # Sudoku (very simple small puzzle)
    with puzzle_tabs[2]:
        st.write("Sudoku — fill the grid. (Demo shows a static puzzle)")
        sudoku = [
            [5,3,0,0,7,0,0,0,0],
            [6,0,0,1,9,5,0,0,0],
            [0,9,8,0,0,0,0,6,0],
            [8,0,0,0,6,0,0,0,3],
            [4,0,0,8,0,3,0,0,1],
            [7,0,0,0,2,0,0,0,6],
            [0,6,0,0,0,0,2,8,0],
            [0,0,0,4,1,9,0,0,5],
            [0,0,0,0,8,0,0,7,9]
        ]
        st.write("Sample Sudoku (static). For demo, mark completed when solved.")
        st.table(sudoku)
        if st.button("Mark Sudoku Completed"):
            c.execute("INSERT INTO puzzles_completed (user_id, puzzle_date, puzzle_type, time_seconds) VALUES (?, ?, ?, ?)",
                      (user['id'], today_str, "sudoku", 300))
            conn.commit()
            st.success("Great! Puzzle completion recorded.")

# ---------------------------
# Analytics page (admin editable / faculty read-only)
# ---------------------------
def analytics_page():
    require_role(['admin','faculty'])
    st.header("Analytics & Reports")
    role = user['role']
    st.write(f"Viewing as **{role}** — admin editable, faculty read-only.")
    # Basic attendance trend
    df = pd.read_sql_query("""SELECT l.lecture_dt as lecture_dt, c.code || ' - ' || c.title as course,
                              SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) as present,
                              COUNT(a.id) as total
                              FROM lectures l
                              LEFT JOIN attendance a ON a.lecture_id=l.id
                              LEFT JOIN courses c ON l.course_id=c.id
                              GROUP BY l.id ORDER BY lecture_dt DESC""", conn)
    if df.empty:
        st.info("No attendance data yet.")
        return
    df['date'] = pd.to_datetime(df['lecture_dt']).dt.date
    agg = df.groupby('date').sum().reset_index()
    agg['pct_present'] = (agg['present'] / agg['total'] * 100).fillna(0)
    fig = px.line(agg, x='date', y='pct_present', title="Daily attendance %")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Department / Course Comparison")
    course_agg = pd.read_sql_query("""SELECT c.code || ' - ' || c.title as course,
                                      SUM(CASE WHEN a.status='present' THEN 1 ELSE 0 END) as present,
                                      COUNT(a.id) as total
                                      FROM attendance a
                                      JOIN lectures l ON a.lecture_id=l.id
                                      JOIN courses c ON l.course_id=c.id
                                      GROUP BY c.id""", conn)
    if not course_agg.empty:
        course_agg['pct'] = (course_agg['present'] / course_agg['total'] * 100).round(1)
        fig2 = px.bar(course_agg, x='course', y='pct', title="Attendance % by Course")
        st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Export")
    if st.button("Export attendance to Excel"):
        df_all = pd.read_sql_query("SELECT a.id, u.name as student, c.code || ' - ' || c.title as course, l.lecture_dt, a.status FROM attendance a JOIN users u ON a.user_id=u.id JOIN lectures l ON a.lecture_id=l.id JOIN courses c ON l.course_id=c.id", conn)
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df_all.to_excel(writer, index=False, sheet_name='attendance')
            writer.save()
        b64 = base64.b64encode(out.getvalue()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="attendance.xlsx">Download attendance.xlsx</a>'
        st.markdown(href, unsafe_allow_html=True)
    # Editing (admin only)
    if role == 'admin':
        st.markdown("---")
        st.subheader("Admin: Quick Edit Attendance")
        att = pd.read_sql_query("SELECT a.id, u.name as student, c.code || ' - ' || c.title as course, l.lecture_dt, a.status FROM attendance a JOIN users u ON a.user_id=u.id JOIN lectures l ON a.lecture_id=l.id JOIN courses c ON l.course_id=c.id", conn)
        st.dataframe(att)
        aid = st.number_input("Attendance ID to edit", min_value=1, step=1)
        new_status = st.selectbox("New status", ["present","absent","late"])
        if st.button("Apply change"):
            c.execute("UPDATE attendance SET status=? WHERE id=?", (new_status, aid))
            conn.commit()
            st.success("Updated attendance record.")

# ---------------------------
# Routing
# ---------------------------
if user:
    if user['role'] == 'admin':
        page = st.selectbox("Choose page", ["Home", "Admin Console", "Analytics", "Puzzles"], index=1)
    elif user['role'] == 'faculty':
        page = st.selectbox("Choose page", ["Home", "Faculty Dashboard", "Analytics"], index=1)
    else:
        page = st.selectbox("Choose page", ["Home", "Daily Puzzles", "My Attendance"], index=0)

    if page == "Admin Console" or page == "Home" and user['role']=='admin':
        admin_page()
    elif page == "Faculty Dashboard" or (page == "Home" and user['role']=='faculty'):
        faculty_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "Daily Puzzles":
        student_page()
    elif page == "My Attendance":
        student_page()
    elif page == "Home" and user['role']=='student':
        student_page()
    else:
        st.write("Page not implemented.")
else:
    st.write("Please login from the sidebar to continue.")
