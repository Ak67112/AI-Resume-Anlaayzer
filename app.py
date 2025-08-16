import os
import json
import re
import csv
from flask import Flask, render_template, request, send_file
import google.generativeai as genai
import PyPDF2
from io import StringIO

app = Flask(__name__)

# Optional upload folder if you want to save PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def extract_resume_text(resume):
    """Extract text from PDF or TXT files"""
    if resume.filename.endswith(".pdf"):
        # Save temporarily if needed
        file_path = os.path.join(UPLOAD_FOLDER, resume.filename)
        resume.save(file_path)
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        # assume text file
        return resume.read().decode("utf-8", errors="ignore")


def clean_json(raw_text):
    """Clean Gemini response to extract valid JSON"""
    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(json)?", "", raw_text)
    raw_text = re.sub(r"```$", "", raw_text)
    raw_text = raw_text.strip()
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        return match.group(0)
    return raw_text


@app.route("/")
def index():
    """Landing page"""
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Resume upload and analysis page"""
    if request.method == "POST":
        job_description = request.form["job_description"]
        resumes = request.files.getlist("resumes")
        results = []

        for resume in resumes:
            text = extract_resume_text(resume)

            prompt = f"""
            You are an AI HR assistant. Compare the following resume against the job description.

            Respond ONLY in valid JSON with this structure:
            {{
              "candidate_name": "string",
              "match_score": int (0-100),
              "match": "Yes" or "No",
              "reason": "string",
              "strengths": ["list of strengths"],
              "gaps": ["list of gaps"],
              "verdict": "string"
            }}

            JOB DESCRIPTION:
            {job_description}

            RESUME:
            {text}
            """

            try:
                response = model.generate_content(prompt)
                raw_text = response.text or ""
                cleaned = clean_json(raw_text)
                data = json.loads(cleaned)

                match_value = str(data.get("match", "No")).lower()
                match_value = "Yes" if match_value.startswith("y") else "No"

                candidate = {
                    "candidate_name": data.get("candidate_name", resume.filename),
                    "match_score": int(data.get("match_score", 0)),
                    "match": match_value,
                    "reason": data.get("reason", "No reason provided"),
                    "strengths": data.get("strengths", []),
                    "gaps": data.get("gaps", []),
                    "verdict": data.get("verdict", "No verdict")
                }

                results.append(candidate)

            except Exception as e:
                results.append({
                    "candidate_name": resume.filename,
                    "match_score": 0,
                    "match": "No",
                    "reason": f"Error parsing response: {e}",
                    "strengths": [],
                    "gaps": [],
                    "verdict": "Could not analyze"
                })

        # Optional: Save selected candidates as CSV
        csv_file = StringIO()
        writer = csv.writer(csv_file)
        writer.writerow(["Candidate Name", "Match Score", "Match", "Reason", "Verdict"])
        for r in results:
            if r["match"] == "Yes":
                writer.writerow([r["candidate_name"], r["match_score"], r["match"], r["reason"], r["verdict"]])
        csv_file.seek(0)

        return render_template("results.html", results=results, csv_data=csv_file.getvalue())

    return render_template("upload.html")


@app.route("/download_csv")
def download_csv():
    """Download CSV of selected candidates"""
    csv_data = request.args.get("csv_data")
    if not csv_data:
        return "No CSV data found", 400
    return send_file(
        StringIO(csv_data),
        mimetype="text/csv",
        as_attachment=True,
        download_name="selected_candidates.csv"
    )


if __name__ == "__main__":
    app.run(debug=True)
