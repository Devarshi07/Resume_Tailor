#!/usr/bin/env python3
"""
AI-Powered Resume Tailoring System
Uses LangChain + Ollama (Llama 3.2 3b) to tailor resumes to job descriptions
"""

import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import difflib
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import argparse

console = Console()

# Pydantic models for structured output
class BulletPointImprovement(BaseModel):
    original: str = Field(description="Original bullet point")
    improved: str = Field(description="Improved bullet point aligned to job description")
    reasoning: str = Field(description="Brief explanation of changes made")
    keywords_added: List[str] = Field(description="Job-relevant keywords incorporated")

class SectionImprovements(BaseModel):
    section_name: str = Field(description="Name of the section (e.g., 'Work Experience', 'Projects')")
    improvements: List[BulletPointImprovement] = Field(description="List of bullet point improvements")

class ResumeAnalysis(BaseModel):
    similarity_score: float = Field(description="Similarity score between resume and job description (0-100)")
    key_missing_skills: List[str] = Field(description="Important skills from job description missing in resume")
    matching_skills: List[str] = Field(description="Skills that match between resume and job description")
    recommendations: List[str] = Field(description="High-level recommendations for alignment")

class AIResumeTailor:
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize the AI Resume Tailor with Ollama LLM"""
        try:
            self.llm = Ollama(model=model_name, temperature=0.3)
            console.print(f"✅ Connected to Ollama model: {model_name}", style="green")
        except Exception as e:
            console.print(f"❌ Error connecting to Ollama: {e}", style="red")
            console.print("Make sure Ollama is running and the model is installed:", style="yellow")
            console.print(f"  ollama pull {model_name}", style="yellow")
            raise
        
        self.setup_chains()
    
    def setup_chains(self):
        """Setup LangChain chains for different tasks"""
        
        # Chain for analyzing resume-job similarity
        analysis_prompt = PromptTemplate(
            input_variables=["resume", "job_description"],
            template="""
            Analyze the similarity between this resume and job description. Be thorough and objective.
            
            RESUME:
            {resume}
            
            JOB DESCRIPTION:
            {job_description}
            
            Provide your analysis in this JSON format:
            {{
                "similarity_score": <score from 0-100>,
                "key_missing_skills": [<list of important missing skills>],
                "matching_skills": [<list of matching skills>],
                "recommendations": [<list of 3-5 high-level recommendations>]
            }}
            
            Be precise and only include skills that are explicitly mentioned or clearly implied.
            """
        )
        
        # Chain for improving bullet points
        improvement_prompt = PromptTemplate(
            input_variables=["bullet_points", "job_description", "section_name"],
            template="""
            You are an expert resume writer. Improve these bullet points from the "{section_name}" section to better align with the job description. 
            
            IMPORTANT RULES:
            1. Keep all achievements and facts truthful - never fabricate experience
            2. Reframe existing accomplishments using job-relevant keywords
            3. Emphasize aspects that match the job requirements
            4. Maintain professional tone and quantifiable metrics
            5. Keep the core meaning and truthfulness intact
            
            BULLET POINTS TO IMPROVE:
            {bullet_points}
            
            JOB DESCRIPTION (for context):
            {job_description}
            
            For each bullet point, provide improvements in this JSON format:
            {{
                "section_name": "{section_name}",
                "improvements": [
                    {{
                        "original": "<original bullet point>",
                        "improved": "<improved version>",
                        "reasoning": "<brief explanation of changes>",
                        "keywords_added": [<relevant keywords added>]
                    }}
                ]
            }}
            
            Focus on making the bullet points more relevant while keeping them truthful.
            """
        )
        
        self.analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
        self.improvement_chain = LLMChain(llm=self.llm, prompt=improvement_prompt)
    
    def extract_sections(self, resume_text: str) -> Dict[str, Dict]:
        """Extract different sections from resume with their formatting"""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            r'(work experience|professional experience|employment|experience)',
            r'(projects?|personal projects?|key projects?)',
            r'(education|academic background)',
            r'(skills?|technical skills?|core competencies)',
            r'(achievements?|accomplishments?|awards?)'
        ]
        
        current_section = None
        current_content = []
        lines = resume_text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            is_section_header = False
            for pattern in section_patterns:
                if re.search(pattern, line_lower) and (len(line.strip()) < 50):
                    is_section_header = True
                    
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = {
                            'content': '\n'.join(current_content),
                            'start_line': len(sections) * 10,  # approximate
                            'bullet_points': self.extract_bullet_points('\n'.join(current_content))
                        }
                    
                    current_section = line.strip()
                    current_content = []
                    break
            
            if not is_section_header and current_section:
                current_content.append(line)
            elif not current_section:
                # Content before first section (header info)
                if 'header' not in sections:
                    sections['header'] = {'content': '', 'bullet_points': []}
                sections['header']['content'] += line + '\n'
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = {
                'content': '\n'.join(current_content),
                'start_line': len(sections) * 10,
                'bullet_points': self.extract_bullet_points('\n'.join(current_content))
            }
        
        return sections
    
    def extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from a text section"""
        bullet_points = []
        lines = text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            # Look for common bullet point indicators
            if (stripped.startswith('•') or stripped.startswith('-') or 
                stripped.startswith('*') or re.match(r'^\d+\.', stripped) or
                (len(stripped) > 20 and any(word in stripped.lower() for word in 
                ['developed', 'managed', 'led', 'created', 'implemented', 'designed', 'achieved']))):
                bullet_points.append(stripped)
        
        return bullet_points
    
    def analyze_resume_job_fit(self, resume: str, job_description: str) -> ResumeAnalysis:
        """Analyze how well the resume fits the job description"""
        console.print("🔍 Analyzing resume-job fit...", style="blue")
        
        try:
            result = self.analysis_chain.run(resume=resume, job_description=job_description)
            # Parse JSON response
            analysis_data = json.loads(result)
            return ResumeAnalysis(**analysis_data)
        except Exception as e:
            console.print(f"⚠️ Analysis error: {e}", style="yellow")
            # Return default analysis
            return ResumeAnalysis(
                similarity_score=50.0,
                key_missing_skills=["Unable to analyze"],
                matching_skills=["Analysis incomplete"],
                recommendations=["Please check your inputs and try again"]
            )
    
    def improve_section_bullets(self, section_name: str, bullet_points: List[str], 
                              job_description: str) -> SectionImprovements:
        """Improve bullet points in a specific section"""
        if not bullet_points:
            return SectionImprovements(section_name=section_name, improvements=[])
        
        console.print(f"✨ Improving {section_name} section...", style="blue")
        
        bullets_text = '\n'.join([f"• {bp}" for bp in bullet_points])
        
        try:
            result = self.improvement_chain.run(
                bullet_points=bullets_text,
                job_description=job_description,
                section_name=section_name
            )
            
            improvement_data = json.loads(result)
            return SectionImprovements(**improvement_data)
        except Exception as e:
            console.print(f"⚠️ Improvement error for {section_name}: {e}", style="yellow")
            # Return original bullet points as fallback
            improvements = [
                BulletPointImprovement(
                    original=bp,
                    improved=bp,
                    reasoning="No changes made due to processing error",
                    keywords_added=[]
                ) for bp in bullet_points
            ]
            return SectionImprovements(section_name=section_name, improvements=improvements)
    
    def rebuild_resume(self, original_resume: str, sections: Dict, improvements: Dict[str, SectionImprovements]) -> str:
        """Rebuild the resume with improvements while maintaining formatting"""
        console.print("🔧 Rebuilding resume with improvements...", style="blue")
        
        # Start with original resume
        rebuilt = original_resume
        
        # Replace improved bullet points section by section
        for section_name, improvement in improvements.items():
            if section_name not in sections:
                continue
                
            section_content = sections[section_name]['content']
            original_bullets = sections[section_name]['bullet_points']
            
            # Create mapping of original to improved bullets
            for imp in improvement.improvements:
                original_bullet = imp.original.lstrip('•-* ').strip()
                improved_bullet = imp.improved
                
                # Find and replace in the section content
                # Try multiple bullet point formats
                patterns = [
                    f"• {re.escape(original_bullet)}",
                    f"- {re.escape(original_bullet)}",
                    f"* {re.escape(original_bullet)}",
                    re.escape(original_bullet)
                ]
                
                for pattern in patterns:
                    if re.search(pattern, section_content, re.IGNORECASE):
                        section_content = re.sub(
                            pattern, 
                            f"• {improved_bullet}", 
                            section_content, 
                            flags=re.IGNORECASE
                        )
                        break
            
            # Replace the entire section in the rebuilt resume
            rebuilt = rebuilt.replace(sections[section_name]['content'], section_content)
        
        return rebuilt
    
    def display_analysis(self, analysis: ResumeAnalysis):
        """Display the resume analysis in a formatted way"""
        console.print("\n" + "="*60, style="cyan")
        console.print("📊 RESUME ANALYSIS RESULTS", style="bold cyan")
        console.print("="*60, style="cyan")
        
        # Similarity score with color coding
        score_color = "green" if analysis.similarity_score >= 70 else "yellow" if analysis.similarity_score >= 50 else "red"
        console.print(f"Overall Similarity Score: {analysis.similarity_score:.1f}%", style=f"bold {score_color}")
        
        # Matching skills
        if analysis.matching_skills:
            console.print(f"\n✅ Matching Skills ({len(analysis.matching_skills)}):", style="green")
            for skill in analysis.matching_skills[:10]:  # Show top 10
                console.print(f"  • {skill}", style="green")
        
        # Missing skills
        if analysis.key_missing_skills:
            console.print(f"\n⚠️ Key Missing Skills ({len(analysis.key_missing_skills)}):", style="yellow")
            for skill in analysis.key_missing_skills[:10]:  # Show top 10
                console.print(f"  • {skill}", style="yellow")
        
        # Recommendations
        if analysis.recommendations:
            console.print(f"\n💡 Recommendations:", style="blue")
            for i, rec in enumerate(analysis.recommendations, 1):
                console.print(f"  {i}. {rec}", style="blue")
    
    def display_improvements(self, improvements: Dict[str, SectionImprovements]):
        """Display the improvements made to each section"""
        console.print("\n" + "="*60, style="magenta")
        console.print("✨ RESUME IMPROVEMENTS", style="bold magenta")
        console.print("="*60, style="magenta")
        
        for section_name, section_improvements in improvements.items():
            if not section_improvements.improvements:
                continue
                
            console.print(f"\n📝 {section_name.upper()}", style="bold magenta")
            console.print("-" * 40, style="magenta")
            
            for imp in section_improvements.improvements:
                console.print(f"\n🔸 Original:", style="dim")
                console.print(f"   {imp.original}", style="dim")
                console.print(f"🔸 Improved:", style="bright_green")
                console.print(f"   {imp.improved}", style="bright_green")
                
                if imp.keywords_added:
                    console.print(f"🔸 Keywords Added: {', '.join(imp.keywords_added)}", style="cyan")
                
                console.print(f"🔸 Reasoning: {imp.reasoning}", style="blue")
                console.print("-" * 30, style="dim")
    
    def process_resume(self, resume_text: str, job_description: str) -> Tuple[str, ResumeAnalysis]:
        """Main function to process and tailor the resume"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Extract sections
            task1 = progress.add_task("Extracting resume sections...", total=None)
            sections = self.extract_sections(resume_text)
            progress.remove_task(task1)
            
            # Step 2: Analyze resume-job fit
            task2 = progress.add_task("Analyzing resume-job compatibility...", total=None)
            analysis = self.analyze_resume_job_fit(resume_text, job_description)
            progress.remove_task(task2)
            
            # Step 3: Improve relevant sections
            task3 = progress.add_task("Tailoring resume sections...", total=None)
            improvements = {}
            
            # Focus on work experience and projects sections
            target_sections = ['work experience', 'professional experience', 'experience', 'projects', 'personal projects']
            
            for section_name, section_data in sections.items():
                if any(target in section_name.lower() for target in target_sections):
                    if section_data['bullet_points']:
                        improvements[section_name] = self.improve_section_bullets(
                            section_name, section_data['bullet_points'], job_description
                        )
            
            progress.remove_task(task3)
            
            # Step 4: Rebuild resume
            task4 = progress.add_task("Finalizing tailored resume...", total=None)
            tailored_resume = self.rebuild_resume(resume_text, sections, improvements)
            progress.remove_task(task4)
        
        # Display results
        self.display_analysis(analysis)
        self.display_improvements(improvements)
        
        return tailored_resume, analysis

def get_user_input() -> Tuple[str, str]:
    """Get resume and job description from user"""
    console.print("🎯 AI-Powered Resume Tailoring System", style="bold blue")
    console.print("This tool will help align your resume to job descriptions using AI\n", style="blue")
    
    # Get resume
    console.print("📄 Please paste your resume text (press Enter twice when done):", style="green")
    resume_lines = []
    empty_line_count = 0
    
    while True:
        line = input()
        if line.strip() == "":
            empty_line_count += 1
            if empty_line_count >= 2:
                break
        else:
            empty_line_count = 0
        resume_lines.append(line)
    
    resume_text = '\n'.join(resume_lines).strip()
    
    if not resume_text:
        console.print("❌ No resume text provided. Exiting.", style="red")
        exit(1)
    
    # Get job description  
    console.print("\n💼 Please paste the job description (press Enter twice when done):", style="green")
    job_lines = []
    empty_line_count = 0
    
    while True:
        line = input()
        if line.strip() == "":
            empty_line_count += 1
            if empty_line_count >= 2:
                break
        else:
            empty_line_count = 0
        job_lines.append(line)
    
    job_description = '\n'.join(job_lines).strip()
    
    if not job_description:
        console.print("❌ No job description provided. Exiting.", style="red")
        exit(1)
    
    return resume_text, job_description

def save_results(tailored_resume: str, analysis: ResumeAnalysis, filename: str = "tailored_resume.txt"):
    """Save the tailored resume and analysis to files"""
    # Save tailored resume
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(tailored_resume)
    
    # Save analysis
    analysis_filename = filename.replace('.txt', '_analysis.json')
    with open(analysis_filename, 'w', encoding='utf-8') as f:
        json.dump(analysis.dict(), f, indent=2)
    
    console.print(f"\n💾 Results saved:", style="green")
    console.print(f"  • Tailored Resume: {filename}", style="green")
    console.print(f"  • Analysis Report: {analysis_filename}", style="green")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI-Powered Resume Tailoring System")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--output", default="tailored_resume.txt", help="Output filename")
    args = parser.parse_args()
    
    try:
        # Initialize the AI Resume Tailor
        tailor = AIResumeTailor(model_name=args.model)
        
        # Get user inputs
        resume_text, job_description = get_user_input()
        
        # Process the resume
        tailored_resume, analysis = tailor.process_resume(resume_text, job_description)
        
        # Display final tailored resume
        console.print("\n" + "="*60, style="green")
        console.print("🎉 TAILORED RESUME", style="bold green")
        console.print("="*60, style="green")
        console.print(tailored_resume)
        console.print("="*60, style="green")
        
        # Save results
        save_results(tailored_resume, analysis, args.output)
        
        console.print(f"\n✅ Resume tailoring completed successfully!", style="bold green")
        
    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style="yellow")
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style="red")
        raise

if __name__ == "__main__":
    main()