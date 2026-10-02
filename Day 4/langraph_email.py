import os
import smtplib

from typing import TypedDict

from email.message import EmailMessage

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1. CHECK API KEY
# ============================================================

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError(
        "GROQ_API_KEY is not set.\n"
        "Set it in Git Bash using:\n"
        'export GROQ_API_KEY="your_api_key"'
    )


# ============================================================
# 2. MODEL
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.3,
    api_key=os.environ["GROQ_API_KEY"]
)


# ============================================================
# 3. AGENT STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str


# ============================================================
# 4. COORDINATOR AGENT
# ============================================================

def coordinator(state: AgentState):

    question = state["question"]

    print("\n================ COORDINATOR ================\n")

    print("Coordinator is analyzing the question...")

    return {
        "question": question
    }


# ============================================================
# 5. RESEARCH AGENT
# ============================================================

def research_agent(state: AgentState):

    question = state["question"]

    print("\n================ RESEARCH AGENT ================\n")

    system_prompt = """
You are a Research Agent.

Analyze the user's question and provide useful research findings.

Cover:
1. Important concepts
2. Requirements
3. Benefits
4. Challenges
5. Important considerations

Give clear and structured information.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ])

    research = response.content

    print(research)

    return {
        "research": research
    }


# ============================================================
# 6. TECHNICAL AGENT
# ============================================================

def technical_agent(state: AgentState):

    question = state["question"]
    research = state["research"]

    print("\n================ TECHNICAL AGENT ================\n")

    system_prompt = """
You are a Kubernetes Technical Architect.

Based on the user's question and research findings, provide
a detailed technical solution.

Cover where relevant:

1. System architecture
2. Kubernetes components
3. Deployment strategy
4. Networking
5. Storage
6. Security
7. Monitoring
8. Scalability
9. High availability
10. Implementation steps

Keep the explanation technically accurate and practical.
"""

    user_prompt = f"""
USER QUESTION:
{question}

RESEARCH FINDINGS:
{research}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    technical = response.content

    print(technical)

    return {
        "technical": technical
    }


# ============================================================
# 7. REPORT AGENT
# ============================================================

def report_agent(state: AgentState):

    question = state["question"]
    research = state["research"]
    technical = state["technical"]

    print("\n================ REPORT AGENT ================\n")

    system_prompt = """
You are a professional technical report writer.

Create a structured final report using the information provided.

The report MUST contain these sections:

1. Executive Summary
2. Research Findings
3. Technical Architecture
4. Implementation Steps
5. Security and Monitoring
6. Conclusion

Write professionally and clearly.

Do not mention that multiple AI agents were used.
"""

    user_prompt = f"""
QUESTION:
{question}

RESEARCH:
{research}

TECHNICAL ANALYSIS:
{technical}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    report = response.content

    print(report)

    return {
        "report": report
    }


# ============================================================
# 8. CREATE PDF
# ============================================================

def create_pdf(question: str, report: str):

    filename = "final_report.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story = []

    # Title
    story.append(
        Paragraph(
            "AI Generated Technical Report",
            title_style
        )
    )

    story.append(Spacer(1, 20))

    # Question
    story.append(
        Paragraph(
            "<b>Question:</b>",
            heading_style
        )
    )

    safe_question = (
        question
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    story.append(
        Paragraph(
            safe_question,
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    # Report
    lines = report.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 8))
            continue

        safe_line = (
            line
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        # Handle markdown headings
        if line.startswith("#"):
            clean_line = line.lstrip("#").strip()

            safe_heading = (
                clean_line
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(
                    safe_heading,
                    heading_style
                )
            )

        else:
            story.append(
                Paragraph(
                    safe_line,
                    normal_style
                )
            )

        story.append(Spacer(1, 5))

    doc.build(story)

    print("\n================================================")
    print("PDF CREATED SUCCESSFULLY")
    print("================================================")
    print(f"File: {os.path.abspath(filename)}")

    return filename


# ============================================================
# 9. EMAIL AGENT
# ============================================================

def email_agent(pdf_file: str):

    print("\n================ EMAIL AGENT ================\n")

    sender_email = os.environ.get("SENDER_EMAIL")
    app_password = os.environ.get("SENDER_APP_PASSWORD")

    if not sender_email:
        raise ValueError(
            'SENDER_EMAIL is not set. In Git Bash run: '
            'export SENDER_EMAIL="your_email@gmail.com"'
        )

    if not app_password:
        raise ValueError(
            'SENDER_APP_PASSWORD is not set. Set your Gmail App Password '
            'in Git Bash as SENDER_APP_PASSWORD.'
        )

    # Ask user for recipient
    recipient_email = input(
        "Enter recipient email: "
    ).strip()

    if not recipient_email:
        print("❌ Recipient email cannot be empty.")
        return

    # Check PDF
    if not os.path.exists(pdf_file):
        print("❌ PDF file not found.")
        return

    # Create email
    msg = EmailMessage()

    msg["Subject"] = "AI Generated Technical Report"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    msg.set_content(
        """Hello,

Please find the AI-generated technical report attached.

Regards,
AI Report Agent
"""
    )

    # Attach PDF
    with open(pdf_file, "rb") as file:

        pdf_data = file.read()

        msg.add_attachment(
            pdf_data,
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_file)
        )

    # Send email using Gmail SMTP with STARTTLS (port 587)
    try:
        print("\\nConnecting to Gmail...")

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        print("\\n================================================")
        print("EMAIL SENT SUCCESSFULLY")
        print("================================================")
        print(f"From      : {sender_email}")
        print(f"To        : {recipient_email}")
        print(f"Attachment: {pdf_file}")
        return True

    except (smtplib.SMTPException, OSError) as e:
        print("\\n❌ EMAIL SENDING FAILED")
        print(f"Error: {e}")
        print("Check your internet connection, firewall/antivirus, Gmail App Password,")
        print("and whether your network allows SMTP on port 587.")
        return False


# ============================================================
# 10. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node(
    "coordinator",
    coordinator
)

graph.add_node(
    "research_agent",
    research_agent
)

graph.add_node(
    "technical_agent",
    technical_agent
)

graph.add_node(
    "report_agent",
    report_agent
)


# ============================================================
# 11. GRAPH FLOW
# ============================================================

graph.add_edge(
    START,
    "coordinator"
)

graph.add_edge(
    "coordinator",
    "research_agent"
)

graph.add_edge(
    "research_agent",
    "technical_agent"
)

graph.add_edge(
    "technical_agent",
    "report_agent"
)

graph.add_edge(
    "report_agent",
    END
)


app = graph.compile()


# ============================================================
# 12. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("================================================")
    print("        AI MULTI-AGENT REPORT SYSTEM")
    print("================================================")

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:

        print("❌ Question cannot be empty.")

    else:

        # Run LangGraph
        result = app.invoke({
            "question": question,
            "research": "",
            "technical": "",
            "report": ""
        })

        # Get final report
        final_report = result["report"]

        # Create PDF
        pdf_file = create_pdf(
            question,
            final_report
        )

        # Send PDF through email agent
        email_agent(pdf_file)

        print("\n================================================")
        print("              PROCESS COMPLETED")
        print("================================================")