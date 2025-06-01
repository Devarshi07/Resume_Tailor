import streamlit as st
import os
import tempfile
from tailor import tailor_resume_file

st.set_page_config(page_title="Resume Tailor", layout="wide")

st.title("Resume Tailoring App")
st.write("Upload your resume (PDF or DOCX) and paste the job description to get a tailored resume.")

resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
job_desc_text = st.text_area("Job Description", height=200)

if st.button("Generate Tailored Resume"):
    if not resume_file:
        st.error("Please upload a resume file.")
    elif not job_desc_text.strip():
        st.error("Please enter a job description.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp:
            tmp.write(resume_file.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Tailoring resume..."):
                tailored_text = tailor_resume_file(tmp_path, job_desc_text)
            st.success("Tailored resume generated!")
            st.text_area("Tailored Resume Output", value=tailored_text, height=400)
            st.download_button(
                label="Download Tailored Resume as Text",
                data=tailored_text,
                file_name="tailored_resume.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error tailoring resume: {e}")
        finally:
            os.remove(tmp_path)
