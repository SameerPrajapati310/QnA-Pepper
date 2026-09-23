# 🌶️ QnA-Pepper

**QnA-Pepper** is a simple AI-powered tool that helps students and learners turn their **PDF study material into a set of useful questions**.

Instead of manually preparing questions from lengthy notes, simply upload a PDF and let QnA-Pepper analyze the content and generate questions for practice, revision, and interview preparation.

![QnA-Pepper](https://github.com/user-attachments/assets/04d478f1-3fa3-466d-87de-072071752390)

---

## ✨ Features

- 📄 **Upload PDF** – Upload your notes, study material, resume, or other PDF documents.
- 🤖 **AI-Powered Question Generation** – Automatically generate questions from the uploaded content.
- 🎓 **Student Friendly** – Useful for exam preparation, revision, and self-assessment.
- 💼 **Interview Preparation** – Convert technical or educational material into interview-style questions.
- 👀 **PDF Preview** – Preview the uploaded PDF directly in the application.
- 📥 **Download Questions** – Download the generated questions for offline practice.
- ⚡ **Simple Interface** – Minimal and easy-to-use UI.

---

## 🎯 Why QnA-Pepper?

Preparing questions manually from a large PDF can be time-consuming.

QnA-Pepper makes the process easier:

```text
        📄 PDF
          │
          ▼
   ┌───────────────┐
   │   QnA-Pepper  │
   │   AI Analysis │
   └───────────────┘
          │
          ▼
   ❓ Generated Questions
          │
          ▼
      📚 Practice
```

Upload your learning material and get a ready-to-use set of questions in a few steps.

---

## 🚀 How It Works

### 1. Upload your PDF

Select a PDF containing your:

- Study notes
- Lecture material
- Technical documentation
- Resume
- Course material
- Interview preparation material

Currently, the application supports PDFs with **up to 10 pages**.

### 2. Analyze the PDF

Once uploaded, QnA-Pepper processes the document and extracts the relevant information.

The extracted content is then used to prepare meaningful questions based on the document.

### 3. Generate Questions

The application generates a set of questions based on the uploaded content.

These questions can be used for:

- 📝 Self-assessment
- 🎓 Exam preparation
- 🔄 Quick revision
- 💻 Technical interview preparation
- 🧠 Knowledge testing

### 4. Download the Questions

Once processing is complete, the generated questions can be downloaded and used for practice.

---

## 🖥️ Application Flow

```text
┌─────────────────────┐
│     Upload PDF      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Validate PDF      │
│   Max 10 Pages      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Analyze PDF      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Generate Questions  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Download Questions  │
└─────────────────────┘
```

---

## 🛠️ Tech Stack

The project uses a lightweight web-based architecture.

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- jQuery
- Font Awesome
- SweetAlert2
- PDF.js

### Backend

The frontend communicates with the backend through two main API endpoints:

```text
POST /upload
```

Used to upload and validate the PDF.

```text
POST /analyze
```

Used to analyze the uploaded PDF and generate the questions.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/QnA-Pepper.git
```

Navigate to the project directory:

```bash
cd QnA-Pepper
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the backend server:

```bash
python app.py
```

Then open the application in your browser:

```text
http://localhost:5000
```

Upload a PDF and start generating questions.

---

## 📌 Example Use Cases

### 🎓 Students

Upload lecture notes and generate questions for revision.

### 💻 Developers

Upload technical documentation and generate questions to test your understanding.

### 💼 Job Seekers

Upload your resume or preparation material and generate potential interview questions.

### 👨‍🏫 Educators

Upload course material and quickly create question sets for students.

---

## 🔮 Future Improvements

Some potential improvements for future versions:

- [ ] Support PDFs larger than 10 pages
- [ ] Generate different difficulty levels
- [ ] Generate MCQs
- [ ] Generate answers along with questions
- [ ] Generate coding questions
- [ ] Generate interview questions by difficulty
- [ ] Support multiple PDFs
- [ ] Question categories
- [ ] Question history
- [ ] User accounts
- [ ] Export questions as PDF/DOCX
- [ ] Interactive quiz mode
- [ ] Score and performance tracking

---

## 🤝 Contributing

Contributions are welcome!

If you have an idea for improving QnA-Pepper:

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/my-feature
```

3. Make your changes
4. Commit your changes

```bash
git commit -m "Add new feature"
```

5. Push the branch

```bash
git push origin feature/my-feature
```

6. Open a Pull Request

---

## 📜 License

This project is open source. Add your preferred license here, such as **MIT License**, if applicable.

---

## 🌶️ QnA-Pepper

> **Upload your PDF. Generate questions. Practice smarter.**

Made to make learning and interview preparation a little easier. 🚀
