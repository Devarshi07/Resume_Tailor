import re
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# Initialize Ollama LLM
llm = Ollama(model="llama3.2-3b")

prompt_template = PromptTemplate(
    input_variables=["job_description", "bullet"],
    template=(
        "You are a resume‐tailoring assistant. Given a job description and one of the applicant’s bullet points,\n"
        "rewrite the bullet so that it aligns more directly with the job description’s language and requirements,\n"
        "without inventing any new experience or changing the core meaning. Preserve the level of seniority and\n"
        "responsibilities exactly as written; only adjust phrasing and keywords.\n\n"
        "Job Description:\n{job_description}\n\n"
        "Original Bullet Point:\n{bullet}\n\n"
        "Rewritten Bullet Point (do not add new details):"
    )
)


def align_bullet(job_desc: str, bullet: str) -> str:
    prompt = prompt_template.format(job_description=job_desc, bullet=bullet)
    rewritten = llm(prompt).strip()
    rewritten = re.sub(r"(?i)^Rewritten Bullet Point:\s*", "", rewritten)
    return rewritten


def align_bullets_to_job(job_desc: str, bullets: list[str]) -> list[str]:
    aligned = []
    for b in bullets:
        m = re.match(r"^([\t ]*(?:[-\*\u2022])\s+)(.*)", b)
        if m:
            prefix, core = m.group(1), m.group(2).strip()
        else:
            prefix, core = "- ", b.strip()
        new_core = align_bullet(job_desc, core)
        aligned.append(f"{prefix}{new_core}")
    return aligned