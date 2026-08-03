import os #access to Python's built in os module, allows interaction with os, reading env vars, working with file paths, etc.
from openai import OpenAI
from dotenv import load_dotenv 
import uuid
import gradio as gr
import uuid
import chromadb
from pprint import pprint
import json
import random
import requests


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("API key is missing.")
else:
    print(OPENAI_API_KEY[:8])
        
client = OpenAI()

#Document
document_overview = """
Lily Lin is a New York–based Senior Project Manager and Vice President specializing in enterprise transformation, technical program management, and product delivery within fintech and hedge-fund environments. She has more than eight years of experience leading complex initiatives from strategy and prioritization through design, implementation, launch, and post-release support. Her work frequently connects engineering, finance, product, operations, executive leadership, vendors, and end users.

She is particularly effective at bringing structure to ambiguous problems, coordinating multiple concurrent workstreams, establishing scalable operating processes, improving visibility through governance and reporting, and keeping technical and business stakeholders aligned. Her core areas of expertise include enterprise transformation, platform modernization, trading technology, operating-model design, process optimization, product requirements, release management, stakeholder communication, workflow automation, data analysis, and AI-enabled project operations.


Communication style: Direct, friendly, warm, supportive.
Make sure to only use factual information about Lily presented above. If you don't know something, just say so.

Additional info: 
"""

document_education = """
I earned a Bachelor of Engineering from The Cooper Union for the Advancement of Science and Art, where I studied from 2013 to 2017 on a four-year full-tuition scholarship. My engineering education gave me a rigorous foundation in analytical thinking, technical problem-solving, systems design, and quantitative reasoning, skills that continue to shape how I approach product, program, and transformation work. While at Cooper Union, I also gained experience developing and maintaining an AP Chemistry course on edX that enrolled more than 5,000 students. Before college, I attended Stuyvesant High School from 2009 to 2013. I was also an iGEM 2014 Gold Medalist and a SASEtank 2017 Finalist.
"""

document_professional_experience=""" 
I have more than eight years of experience leading product, program, and transformation initiatives across fintech, electronic trading, hedge funds, data analytics, and enterprise operations. My career has progressed from hands-on technical roles in engineering, data, and automation to product ownership, technical project management, and large-scale transformation leadership. Throughout this progression, I have developed a strong ability to translate business needs into executable plans, align technical and nontechnical stakeholders, improve operating processes, and deliver measurable business outcomes.

I currently serve as Senior Program Manager and Vice President of Transformation at Tradeweb, where I manage program delivery for the company’s largest finance transformation initiative. I coordinate more than 100 internal stakeholders and external partners across discovery, design, and execution. I have established governance frameworks using Jira, executive reporting, and delivery dashboards to improve visibility, accountability, and alignment across interconnected workstreams. I have also operationalized ChatGPT to automate documentation and support the generation of thousands of project artifacts, increasing the speed and scalability of project operations.

At Tradeweb, I also led a department-wide SharePoint migration under aggressive timelines. I helped define the migration strategy, develop training programs, improve access to organizational knowledge, and enable more efficient AI-supported workflows. In addition, I serve as a trusted resource across the Finance organization, supporting onboarding, project management best practices, and the adoption of tools such as Jira, Monday.com, and SharePoint. This work has strengthened my expertise in enterprise transformation, organizational change, knowledge management, AI adoption, program governance, and cross-functional execution.

Before Tradeweb, I worked at Citadel as a Senior Technical Project/Product Manager. I led the end-to-end delivery of initiatives for a proprietary trading platform, managing work from prioritization and roadmap development through engineering, testing, deployment, and launch. I partnered with senior technical and business leaders to evaluate tradeoffs, manage risks, resolve dependencies, and coordinate multiple concurrent workstreams.

One of my major Citadel initiatives involved coordinating development changes, quality assurance testing, configurations, and deployment activities to change a trading system’s restart schedule. The completed initiative eliminated 40 minutes of daily system downtime during a period of high trading activity. I also directed a vendor integration across engineering, QA, and user-testing teams that reduced trader workflow time by 50 percent. These projects required careful coordination in a high-performance environment where system reliability, execution speed, and operational efficiency were critical.

From 2019 to 2022, I worked at Millennium as a Product Owner for equities trading applications. I shipped a desktop trading application that consolidated multiple system instances into a single global blotter. The product significantly improved order-entry efficiency and reduced onboarding time for new traders by half. I also led day-to-day product operations for an iOS equities trading application, helping the team triple its output per release cycle.

At Millennium, I created a standardized release playbook that reduced deployment failures and doubled release frequency. I partnered with portfolio managers to translate business needs into product requirements, and the resulting enhancements increased mobile application usage by 75 percent. My responsibilities covered the full product lifecycle, including requirements writing, feature wireframes, prioritization, collaboration with engineering and QA, user training, feedback collection, release coordination, and post-launch production support.

Earlier, I served as a Product Owner at MarketAxess, where I managed the development of portfolio-trading functionality from concept through launch. The solution reduced order-entry time for large baskets by 75 percent and increased the maximum supported trade size by 500 percent. I also established a faster development process that allowed high-priority enhancements to be delivered in two weeks rather than waiting for the standard three-month release cycle. The process exceeded client expectations and was later adopted by other teams.

My MarketAxess responsibilities also included producing functional requirements, conducting SQL-based analysis of production issues and trading behavior, presenting product demonstrations, and resolving post-release problems. My demonstrations reduced production-related questions by more than 50 percent, reinforcing the value of clear user communication, training, and release preparation.

My earlier experience gave me a foundation in analytics, software automation, and engineering. At JUST Capital, I helped develop business intelligence dashboards and analyzed data from thousands of surveys to generate actionable insights. I also created ETL processes in GoodData CloudConnect to model, relate, visualize, and report data.
At Yext, I developed a Python-based daily task notifier that reduced manual review by 60 percent and wrote SQL scripts that identified time-sensitive bugs affecting VIP clients, improving response times by more than 90 percent. At the Air Force Institute of Technology, I contributed to microelectronic verification research by developing Python-based gate-identification algorithms and conducting simulations using Cadence Virtuoso and Spectre. I also worked as a MOOC developer at Cooper Union, organizing and maintaining an AP Chemistry course on edX that enrolled more than 5,000 students.

Across these roles, I have developed expertise in financial technology, electronic trading, product development, enterprise transformation, technical program management, workflow automation, process optimization, AI-enabled operations, and stakeholder leadership. My strongest contribution is often connecting strategy with execution: defining what needs to happen, creating structure around complex work, aligning the people responsible for delivery, and ensuring that the final outcome produces measurable value."""

#chunking function
def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    BOUNDARIES = ["\n\n", "\n", ". ", "? ", "! ", " "]

    def find_natural_boundary(start: int, end: int) -> int:
        midpoint = start + (chunk_size // 2)
        for boundary in BOUNDARIES:
            pos = text.rfind(boundary, midpoint, end)
            if pos != -1:
                return pos + len(boundary)
        return end
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = find_natural_boundary(start, end)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks

documents = [
    {"text":document_overview, "source": "Overview"},
    {"text":document_education, "source": "Education"},
    {"text":document_professional_experience, "source": "Professional Experience"}
]

chunks = []
ids = []
metadatas = []

for doc in documents:
    chunks_ = split_text_into_chunks(doc["text"], chunk_size=150, overlap=50)
    ids_ = [str(uuid.uuid4()) for _ in range(len(chunks_))]
    metadatas_= [{"source":doc["source"], "chunk_index": i} for i in range(len(chunks_))]

    chunks.extend(chunks_)
    ids.extend(ids_)
    metadatas.extend(metadatas_)

#print for logs
print(f"Created {len(chunks)} chunks:\n")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} (ID: {ids[i]}, Source: {metadatas[i]['source']}, Index: {metadatas[i]['chunk_index']}, Length: {len(chunk)}):")
    print(chunk)
    print()

#generate embeddings
response = client.embeddings.create(
    model = "text-embedding-3-small",
    input = chunks
)

embeddings = [item.embedding for item in response.data]

#initialize chromaDB client for persistent storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="digital_twin")
if collection.get()["ids"]:
    collection.delete(collection.get()["ids"])

collection.add( #parameters for the collection
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas
)

pprint(collection.get())

#-----------------
# Tools
#-----------------
tools = []
#All Pushover notif code
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def send_notification(message: str):
    if pushover_user is None or pushover_token is None:
        return "Notification failed"
    payload = {"user":pushover_user, "token":pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

send_notification_function = {
    "name": "send_notification",
    "description": "Sends a push notif to the real world version of you via Pushover. Use this if the user needs to alert the real world version of you",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The notification message to send to the user's device"
            }
        },
        "required": ["message"]
    }
}

tools.append ({"type":"function", "function":send_notification_function})

#Dice rolling functionality
def dice_roll():
    result = random.randint(1,6)
    return result

roll_dice_function = {
    "name": "dice_roll",
    "description": "Simulates rolling a single six sided die and returns the result. Use this when the user wants to roll a die for games, deisions, or random number generator",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
        },
    }

tools.append({"type":"function", "function":roll_dice_function})

#Function to handle LLM tool calling
def handle_tool_call(tool_calls):
    tool_results =[]
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if function_name == "send_notification":
            send_notification(args["message"])
            content = f"Notification sent: {args['message']}"
        elif function_name == "dice_roll":
            content = f"Rolled: {dice_roll()}"
            #send_notification(args["message"])
        else:
            content = f"Unknown function:{function_name}"

        tool_call_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
    }

        tool_results.append(tool_call_result)
        
    return tool_results

#system message
system_message = """You are a digital career twin of Lily. When people talk to you, you respond as Lily - in first person, using her voice, and knowledge.
Start with a warm and energetic greeting to the person, asking them what's on their mind today. 
If any personal questions are asked regarding Lily, simply say that you do not possess this information. The only exception is if this is actively offered in the context.
Here's the information about Lily to help you embody the professional version of her:
#main response functon"""

#with response function
def respond_ai(message,history): #the function that Gradio calls everytime user sends a message. Passes the message and the history
    #RAG: Embed query using same model for chunks
    response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = [message] #input is always a list
    )

    query_embedding = response.data[0].embedding

    #RAG: search chromadb
    results = collection.query( #queries for the 3 closest matches, applies fuzzy matching is not exact match
        query_embeddings=[query_embedding],
        n_results=3
    )

    #RAG: stitch retrieved chunks together to create the context for the response
    context = "\n---\n".join(results["documents"][0])

    #print logs for debugging
    print("\n==========================\n")
    print(f"User message:\n{message}\n")
    print("***Retrieved Chunks:")
    for a, b in zip(results["documents"][0], results["metadatas"][0]):
        print(f"<<Document: {b['source']} --Chunk {b['chunk_index']}>>\n{a}\n")
    system_message_enhanced = system_message + "\n\nContext:\n" + context    

    #build messages for this turn
    messages = [{"role":"system", "content":system_message_enhanced}] + history + [{"role":"user", "content":message}]

    #call LLM
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages = messages,
        tools=tools
    )
    
    message=response.choices[0].message

    #check if model wants to call a tool
    while message.tool_calls:
        from pprint import pprint
        pprint (message.tool_calls)
        #handle tool call, add info about tool call response to 'context', invoke LLM one more time to get its new updated response
        tool_result = handle_tool_call(message.tool_calls)
        messages.append(message)
        messages.extend(tool_result)
        #args = json.loads(tool_call.function.arguments)
        #send_notification(args["message"])
        #print(f"Sent notification:{args['message']}")
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools
        )
        message=response.choices[0].message


    return(message.content)

#launch gradio
#gr.ChatInterface(
#    fn=respond_ai).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))  

if __name__ == "__main__":
    demo = gr.ChatInterface(
    fn=respond_ai,
    type="messages",
)

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False,
    )