import os
import sys
from typing import TypedDict

# Ensure UTF-8 output on Windows console
sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ==============================
# NEW: PDF IMPORTS
# ==============================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3,
    api_key=os.environ.get("GROQ_API_KEY"),
)


# Shared state for all agents
class AgentState(TypedDict):
    question: str
    research: str
    technical: str
    report: str


# Coordinator node
def coordinator(state: AgentState):
    print("\n[Coordinator] Starting workflow")

    return {
        "research": "",
        "technical": "",
        "report": ""
    }


# Agent 1: Research Agent
def research_agent(state: AgentState):
    print("\n[Research Agent] Working...")

    prompt = f"""
    Analyze the following question as a Research Agent.

    Identify:
    - Important concepts
    - Requirements
    - Benefits
    - Challenges

    Question: {state['question']}
    """

    response = llm.invoke([
        SystemMessage(content="You are an IT Research Agent."),
        HumanMessage(content=prompt)
    ])

    return {"research": response.content}


# Agent 2: Technical Agent
def technical_agent(state: AgentState):
    print("\n[Technical Agent] Working...")

    prompt = f"""
    You are a Kubernetes Technical Architect.

    Based on the research, propose a technical solution.

    Include:
    - Architecture
    - Kubernetes components
    - Deployment approach
    - Security
    - Monitoring

    User question:
    {state['question']}

    Research findings:
    {state['research']}
    """

    response = llm.invoke([
        SystemMessage(content="You are a Kubernetes expert."),
        HumanMessage(content=prompt)
    ])

    return {"technical": response.content}


# Agent 3: Report Agent
def report_agent(state: AgentState):
    print("\n[Report Agent] Working...")

    prompt = f"""
    Prepare a structured technical report.

    Include:
    1. Executive summary
    2. Research findings
    3. Technical architecture
    4. Implementation steps
    5. Conclusion

    User question:
    {state['question']}

    Research:
    {state['research']}

    Technical solution:
    {state['technical']}

    Do not invent unsupported facts.
    """

    response = llm.invoke([
        SystemMessage(content="You are a Technical Report Agent."),
        HumanMessage(content=prompt)
    ])

    return {"report": response.content}


# Build LangGraph
graph = StateGraph(AgentState)

# Register nodes
graph.add_node("coordinator", coordinator)
graph.add_node("research", research_agent)
graph.add_node("technical", technical_agent)
graph.add_node("report", report_agent)

# Define workflow
graph.add_edge(START, "coordinator")
graph.add_edge("coordinator", "research")
graph.add_edge("research", "technical")
graph.add_edge("technical", "report")
graph.add_edge("report", END)

# Compile graph
app = graph.compile()


# ============================================================
# NEW: FUNCTION TO SAVE ALL AGENT INFORMATION INTO PDF
# ============================================================

def save_report_pdf(result, question):

    filename = "multi_agent_report.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    story = []

    # Title
    story.append(
        Paragraph(
            "Multi-Agent Technical Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    # User Question
    story.append(
        Paragraph(
            "User Question",
            styles["Heading1"]
        )
    )

    story.append(
        Paragraph(
            question,
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 20))

    # Research Agent
    story.append(
        Paragraph(
            "1. Research Agent",
            styles["Heading1"]
        )
    )

    for line in result["research"].split("\n"):

        if line.strip():

            story.append(
                Paragraph(
                    line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"),
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1, 5))

    story.append(Spacer(1, 20))

    # Technical Agent
    story.append(
        Paragraph(
            "2. Technical Agent",
            styles["Heading1"]
        )
    )

    for line in result["technical"].split("\n"):

        if line.strip():

            story.append(
                Paragraph(
                    line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"),
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1, 5))

    story.append(Spacer(1, 20))

    # Final Report
    story.append(
        Paragraph(
            "3. Final Report",
            styles["Heading1"]
        )
    )

    for line in result["report"].split("\n"):

        if line.strip():

            story.append(
                Paragraph(
                    line.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;"),
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1, 5))

    # Build PDF
    doc.build(story)

    print(
        f"\n[PDF] Report saved successfully: {os.path.abspath(filename)}"
    )


# Execute application
if __name__ == "__main__":
    question = input("Enter your question: ")

    result = app.invoke({
        "question": question,
        "research": "",
        "technical": "",
        "report": ""
    })

    print("\n========== FINAL REPORT ==========")
    print(result["report"])

    # ========================================================
    # NEW: SAVE ALL AGENT INFORMATION TO PDF
    # ========================================================

    save_report_pdf(result, question)
