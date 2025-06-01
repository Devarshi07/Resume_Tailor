from resume_loader import load_resume_text
from bullet_utils import extract_section, replace_section, extract_bullets, replace_bullets
from llm_utils import align_bullets_to_job


def tailor_resume_file(resume_path: str, job_description: str) -> str:
    resume_text = load_resume_text(resume_path)
    tailored = resume_text

    for section_name in ["Work Experience", "Projects"]:
        original_section_body = extract_section(tailored, section_name)
        if not original_section_body.strip():
            continue
        old_bullets = extract_bullets(original_section_body)
        if not old_bullets:
            continue
        new_bullets = align_bullets_to_job(job_description, old_bullets)
        new_section_body = replace_bullets(original_section_body, new_bullets)
        tailored = replace_section(tailored, section_name, new_section_body)
    return tailored