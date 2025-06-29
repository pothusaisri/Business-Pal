import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import re
import os
import fitz  # PyMuPDF
from io import StringIO
from serpapi.google_search import GoogleSearch
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any, Literal
from dotenv import load_dotenv
import pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from datetime import datetime, timedelta
import time
import hashlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    FAISS_AVAILABLE = True
except ImportError:
    st.warning("FAISS not installed. Please run: pip install faiss-cpu")
    FAISS_AVAILABLE = False
# Load environment variables
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  

pc = Pinecone(api_key= PINECONE_API_KEY)


def get_llm(model_name):
    """Return appropriate LLM based on selected model name"""
    if model_name.startswith("gpt"):
        return ChatOpenAI(
            model=model_name, 
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
    elif model_name == "gemini-1.5-flash":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=GOOGLE_API_KEY
        )
    elif model_name == "deepseek-reasoner":
        return ChatDeepSeek(model="deepseek-reasoner", 
                            temperature=0.3,
                            api_key=DEEPSEEK_API_KEY)
    else:
        # Fallback to GPT-4o
        return ChatOpenAI(model="gpt-4o", temperature=0.3)


# ======================== STATE MANAGEMENT ========================
def init_session_state():
    session_defaults = {
        "business_context": "",
        "business_data": None,
        "messages": [],
        "model_name": "gpt-4o",
        "specifications": "",
        "locations": [],
        "exclude": [],
        "notes": "",
        "context_file": None,
        "context_text": "",
        "selected_metrics": [],
        "show_supplier_form": False,
        "pinecone_index": None,  # Replace chroma_collection
        "context_namespace": None,
        "context_chunks": [],
        "web_cache": {},
        "context_set_time": None,
        "context_set": False,
        "prev_model": "gpt-4o"
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ======================== TOOLS ========================
def extract_text_from_file(uploaded_file):
    """Extract text from uploaded file (TXT or PDF)"""
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    elif uploaded_file.type == "application/pdf":
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    return ""

#-======================== DB INITIALIZATION ========================
def create_pinecone_index(context_text):
    """Create vector index from business context using Pinecone"""
    if not context_text or len(context_text) < 500:
        return None, None
    
    namespace = hashlib.md5(context_text.encode()).hexdigest()

    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # outputs 1536-dim vectors
        api_key=OPENAI_API_KEY
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(context_text)

    index_name = "business-context"
    
    # 🔥 Force-delete if existing index has wrong dimension
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
        time.sleep(5)  # Wait a few seconds to avoid race conditions

    pc.create_index(
        name=index_name,
        dimension=1536,  # ✅ Must match embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    vector_store = PineconeVectorStore.from_texts(
        chunks,
        embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    return vector_store, namespace

def retrieve_relevant_context(query, k=3):
    """Retrieve top k relevant context chunks using Pinecone"""
    if not st.session_state.pinecone_index:
        return st.session_state.business_context[:3000]
    
    try:
        # Use the vector store to do similarity search
        results = st.session_state.pinecone_index.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        st.error(f"Context retrieval error: {str(e)}")
        return st.session_state.business_context[:3000]

# === Web Search with Caching ===
def search_web(query: str) -> str:
    """Uses SerpAPI to search for latest web trends."""
    # Cache management
    cache_key = f"{query[:100]}"
    cached = st.session_state.web_cache.get(cache_key)
    
    # Return cached result if exists and not expired
    if cached and datetime.now() - cached["timestamp"] < timedelta(hours=1):
        return cached["results"]
    
    # Perform fresh search
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google",
        "num": 5
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    web_context = []
    for item in results.get("organic_results", []):
        title = item.get("title", "Untitled")
        link = item.get("link", "#")
        snippet = item.get("snippet", "No description available")
        web_context.append(f"### {title}\n**Source**: [{link}]({link})\n{snippet}\n")
    
    web_content = "\n".join(web_context)
    
    # Update cache (limit to 50 entries)
    if len(st.session_state.web_cache) > 50:
        # Remove oldest entry
        oldest_key = min(st.session_state.web_cache, key=lambda k: st.session_state.web_cache[k]["timestamp"])
        del st.session_state.web_cache[oldest_key]
    
    st.session_state.web_cache[cache_key] = {
        "results": web_content,
        "timestamp": datetime.now()
    }
    
    return web_content



# ======================== AGENT DEFINITIONS ========================
class BaseAgent:
    def __init__(self, model_name="gpt-4o"):
        self.llm = get_llm(model_name)
        
    def get_web_context(self, query, business_context):
        """Get web context relevant to both query and business context"""
        search_query = f"{query} in context of {business_context[:100]}"
        return search_web(search_query)
    
    def get_business_context(self, query):
        """Get relevant business context chunks"""
        return retrieve_relevant_context(query)

class SupplierAgent(BaseAgent):
    def search_suppliers(self, query, specifications, locations, exclude, notes, context):
        """Supplier research with enhanced capabilities"""
        # Build search query
        search_query = f"Find suppliers for {query} {specifications}"
        if locations:
            search_query += f" in {', '.join(locations)}"
        
        # Add urgency markers
        if "urgent" in notes.lower() or "fast" in notes.lower():
            search_query += " with fast delivery"
        
        # Configure search parameters
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": SERPAPI_API_KEY,
            "num": 15,
            "gl": "us",
            "hl": "en"
        }
        
        # Execute search
        results = GoogleSearch(params).get_dict()
        raw_suppliers = []
        
        # Process organic results
        for item in results.get("organic_results", []):
            raw_suppliers.append({
                "name": item.get("title", "Untitled"),
                "website": item.get("link", "#"),
                "snippet": item.get("snippet", "No description available")
            })
            
        # Enrich and filter suppliers
        return self.enrich_suppliers(raw_suppliers)[:10]  # Return top 10 results

    def enrich_suppliers(self, raw_suppliers):
        """Enrich supplier data and extract locations with improved accuracy"""
        enriched_suppliers = []
        
        # Improved location extraction patterns
        location_patterns = [
            r'headquartered in ([A-Z][a-zA-Z\s]+)',
            r'based in ([A-Z][a-zA-Z\s]+)',
            r'located in ([A-Z][a-zA-Z\s]+)',
            r'manufactur(?:ed|ing) in ([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+), [A-Z]{2,3} (?:manufacturer|supplier|company)',
            r'([A-Z][a-zA-Z\s]+)-based (?:company|firm)'
        ]
        
        # Common non-supplier indicators (government, educational, etc.)
        non_supplier_indicators = [
            "government", "university", "research", "study", "how to", "guide",
            "encyclopedia", "wikipedia", "encyclopaedia", "sciencedirect",
            "energy.gov", "ucs.org", "blog", "article", "pdf", "faq"
        ]
        
        for supplier in raw_suppliers:
            name = supplier['name']
            snippet = supplier['snippet']
            website = supplier['website']
            
            # Skip non-supplier results
            if any(indicator in name.lower() or indicator in snippet.lower() 
                   for indicator in non_supplier_indicators):
                continue
            
            # Extract location using multiple patterns
            location = "Global"
            for pattern in location_patterns:
                match = re.search(pattern, snippet, re.IGNORECASE)
                if match:
                    location = match.group(1).strip()
                    break
            
            # Classify supplier type
            supplier_type = "Unknown"
            if "manufactur" in snippet.lower():
                supplier_type = "Manufacturer"
            elif "supplier" in snippet.lower():
                supplier_type = "Supplier"
            elif "solution" in snippet.lower():
                supplier_type = "Solution Provider"
            elif "distributor" in snippet.lower():
                supplier_type = "Distributor"
            
            enriched_suppliers.append({
                "name": name,
                "website": website,
                "hq": location,
                "description": snippet,
                "type": supplier_type
            })
        
        return enriched_suppliers # Return top 10 results
    
class NewsletterAgent(BaseAgent):
    def generate_copy(self, query, context):
        """Generate marketing email/newsletter from user prompt"""
        business_context = self.get_business_context(query)
        web_context = self.get_web_context(f"email marketing trends for {query}", business_context)
 
        prompt = ChatPromptTemplate.from_template(
            """You are Marketing email/newsletter Agent. Based on this user request:
 
            "{query}"
 
            And business context:
 
            {context}
 
            And current email marketing trends from web:
 
            {web_context}
 
            Write a clear, catchy but professional marketing email with:
 
            - Subject line
            - Email body
 
            Format:
 
            ## Subject:
            [subject line here]
 
            ## Email Body:
 
            [well formatted email body here]
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "web_context": web_context})


class PricingAgent(BaseAgent):
    def analyze_pricing(self, query, context):
        """Comprehensive pricing strategy analysis"""
        business_context = self.get_business_context(query)
        web_context = self.get_web_context(f"pricing strategies for {query}", business_context)

        prompt = ChatPromptTemplate.from_template(
            """
            "You are PricingStrategistAgent. Given our BUSINESS PROFILE (including cost structure and positioning), suggest:\n"
            "• MSRP to hit target margins\n"
            "• Promotional / markdown strategies\n\n"
            As a pricing strategist, develop recommendations for:
            {query}
            "=== BUSINESS PROFILE ===\n"
            "{context}" and using Market Context from Web {web_context}
            - **Risk Mitigation**: [strategies]"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "web_context": web_context})
    
class CompetitorAgent(BaseAgent):
    def analyze_competitors(self, query, context):
        """Analyze competitors and whitespace opportunities"""
        # Get relevant web context
        focused_context = self.get_business_context(query)
        web_context = self.get_web_context(f"competitors for {query}", focused_context)
        
        prompt = ChatPromptTemplate.from_template(
            """
            You are MarketResearchAgent. Given the BUSINESS PROFILE below and market context, identify:
            1. 3-5 whitespace opportunities (untapped market areas)
            2. Top 5 competitors & their core offerings
            3. Suggested price brackets
            4. Competitive positioning recommendations
            
            === BUSINESS PROFILE ===
            {context}
            
            === MARKET CONTEXT ===
            {web_context}
            
            Format your response in markdown:
            ## 🏆 Competitor Analysis for {query}            
            ### 🕳️ Whitespace Opportunities
            1. [Opportunity 1] - [Why it's valuable]
            2. [Opportunity 2] - [Why it's valuable]
            
            ### 🥊 Top Competitors
            | Competitor | Core Offerings | Price Range | Market Position |
            |------------|----------------|-------------|-----------------|
            | [Name] | [Key products/services] | [$$$] | [Leader/Challenger/Niche] |
            
            ### 💡 Strategic Recommendations
            - [Positioning strategy 1]
            - [Positioning strategy 2]
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "web_context": web_context})


class TrendAgent(BaseAgent):
    def analyze_trends(self, query, context):
        """Market trend analysis with forecasting"""

        focused_context = self.get_business_context(query)
        web_context = self.get_web_context(f"market trends for {query}", focused_context)

        prompt = ChatPromptTemplate.from_template(
            """As a market analyst, identify key trends for:
            Industry: {query}
            Business Context: {context}
            Market Context from Web : {web_context}
            
            Provide in markdown format:
            ## 📈 Market Trends
            ### Emerging Trends
            1. [Trend 1]
            2. [Trend 2]
            3. [Trend 3]
            
            ### Impact Analysis
            | Trend | Opportunity Level | Business Impact |
            |-------|-------------------|-----------------|
            | [Trend] | [High/Medium/Low] | [Impact description] |
            
            ### Strategic Recommendations
            - [Initiative 1]
            - [Initiative 2]"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "context": context, "web_context": web_context})
    
class GeneralAgent(BaseAgent):
    def answer_question(self, query, business_context):
        """Handle general business questions with web and business context"""
        # Get relevant web context
        business_context = self.get_business_context(query)
        web_context = self.get_web_context(query, business_context)
        
        prompt = ChatPromptTemplate.from_template(
            """You are an AI Business Builder assistant. Your role is to help entrepreneurs and business leaders 
            make informed decisions. Answer the user's question using the business context and current market insights.

            ### Business Context
            {business_context}

            ### Market Insights (Web Search)
            {web_context}

            ### User Question
            {query}
            If the question is not clear or irrelevant then respond generally the llm answer the question in a way that is helpful to the user."""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "query": query, 
            "business_context": business_context,
            "web_context": web_context
        })

# ======================== LANGGRAPH AGENTS ========================
class SupervisorState(TypedDict):
    query: str
    context: str
    agent_type: Literal["supplier", "pricing", "trend", "business_metrics", "general", "competitor", "newsletter", "unknown"]
    specifications: Optional[str]
    locations: Optional[List[str]]
    exclude: Optional[List[str]]
    notes: Optional[str]
    results: Any
    feedback: str
    needs_data: bool

class BusinessMetricsState(TypedDict):
    topic: str
    historic: str
    web: str
    insights: str
    charts: Dict[str, Any]
    key_metrics: Dict[str, Any]

def create_supervisor_workflow():
    """LangGraph workflow for supervisor agent"""
    workflow = StateGraph(SupervisorState)
    
    def classify_intent(state: SupervisorState):
        prompt = f"""Classify this business query into exactly one category: {state['query']}
        
        Categories:
        - supplier: Sourcing vendors, manufacturers, supply chain partners
        - pricing: Product pricing, cost analysis, pricing strategies
        - trend: Market trends, industry forecasts, emerging developments
        - competitor: Competitor analysis, whitespace opportunities, competitive benchmarking
        - business_metrics: Business performance, KPIs, data analysis, historical metrics
        - newsletter: Marketing emails, newsletters, promotional content
        - general: General business advice, strategy, or other non-specific queries
        
        Respond ONLY with the category name.
        """
        
        llm = get_llm(st.session_state.model_name)
        response = llm.invoke([
            SystemMessage(content="You are a business intelligence routing specialist"),
            HumanMessage(content=prompt)
        ])
        
        state["agent_type"] = response.content.lower().strip()
        return state
    

    def route_node(state: SupervisorState):
        """Route to appropriate agent based on classification"""
        business_data = st.session_state.get("business_data")
        business_data_empty = business_data is None or business_data.empty
        
        if state["agent_type"] == "supplier":
            if not state.get("specifications") or not state.get("locations"):
                state["feedback"] = "Need more details for supplier search"
                state["needs_data"] = True
                st.session_state.show_supplier_form = True
                return state
            return {**state, "results": "SUPPLIER_AGENT"}
            
        elif state["agent_type"] == "pricing":
            return {**state, "results": "PRICING_AGENT"}
        
        elif state["agent_type"] == "competitor":
            return {**state, "results": "COMPETITOR_AGENT"}
            
        elif state["agent_type"] == "trend":
            return {**state, "results": "TREND_AGENT"}
        
        elif state["agent_type"] == "newsletter":
            return {**state, "results": "NEWSLETTER_AGENT"}
            
        elif state["agent_type"] == "business_metrics":
            if business_data_empty:
                state["feedback"] = "Need business metrics data"
                state["needs_data"] = True
                return state
            return {**state, "results": "BUSINESS_METRICS_AGENT"}
            
        elif state["agent_type"] == "general":
            return {**state, "results": "GENERAL_AGENT"}
            
        else:
            # Route to general agent as fallback
            state["agent_type"] = "general"
            return {**state, "results": "GENERAL_AGENT"}
    
    workflow.add_node("classify", classify_intent)
    workflow.add_node("route", route_node)
    
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "route")
    workflow.add_edge("route", END)
    
    return workflow.compile()

def create_business_metrics_workflow():
    """LangGraph workflow for business metrics agent"""
    # Create prompt template
    trend_prompt = PromptTemplate.from_template("""
    You are a predictive AI agent analyzing business trends.
    Given the historical data and the latest trends from the web, provide:
    1. A summary of historical performance.
    2. Key trend shifts.
    3. Forecasted developments.
    4. Strategic insights & suggested actions.

    Historical Data:
    {historic}

    Recent Web Trends:
    {web}

    Respond in a markdown bullet-point format.
    """)
    
    # Create chain
    llm = get_llm(st.session_state.model_name)
    
    # Create chain
    trend_chain = LLMChain(
        llm=llm,
        prompt=trend_prompt
    )
    
    # Define nodes
    def fetch_web_trends(state: BusinessMetricsState):
        state['web'] = search_web(state['topic'])
        return state

    def analyze_trends(state: BusinessMetricsState):
        state['insights'] = trend_chain.run(historic=state['historic'], web=state['web'])
        return state
    
    def generate_charts_and_metrics(state: BusinessMetricsState):
        """Generate charts and key metrics from historical data"""
        try:
            # Convert JSON back to DataFrame
            df = pd.read_json(state['historic'])
            
            # Validate DataFrame
            if df.empty:
                state['error'] = "No data available in the uploaded file"
                return state
                
            if "Date" not in df.columns:
                state['error'] = "Data must contain a 'Date' column"
                return state
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                state['error'] = "No numeric metrics found in the data"
                return state
            
            # Use LLM to select the most appropriate metrics for trend analysis
            llm = get_llm(st.session_state.model_name)
            prompt = f"""
            You are a business intelligence analyst. Given a dataset with the following numeric columns:
            {', '.join(numeric_cols)}
            
            And the user's question: "{state['topic']}"
            
            Select the 2-5 most appropriate metrics for trend analysis visualization (line charts) 
            that would provide the most valuable insights for this query. Consider:
            - Business relevance to the query
            - Suitability for trend visualization
            - Potential to show meaningful patterns over time
            
            Return ONLY a comma-separated list of the selected metric names.
            Example: Revenue,Active Users,Conversion Rate
            """
            
            response = llm.invoke([
                SystemMessage(content="You are a data visualization expert"),
                HumanMessage(content=prompt)
            ])
            
            # Parse the LLM response
            selected_metrics = [m.strip() for m in response.content.split(",")]
            selected_metrics = [m for m in selected_metrics if m in numeric_cols]
            
            # Fallback if LLM selection fails
            if not selected_metrics:
                selected_metrics = numeric_cols[:min(3, len(numeric_cols))]
            
            # Generate charts
            charts = {}
            for metric in selected_metrics:
                fig = px.line(
                    df, 
                    x="Date", 
                    y=metric, 
                    title=f"{metric} Trend",
                    markers=True
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Date",
                    yaxis_title=metric,
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                charts[metric] = fig
            
            # Calculate key metrics
            key_metrics = {}
            for metric in selected_metrics:
                # Convert to numeric if needed
                if not np.issubdtype(df[metric].dtype, np.number):
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
                    df = df.dropna(subset=[metric])
                    
                if df[metric].empty:
                    continue
                    
                start_val = df[metric].iloc[0]
                end_val = df[metric].iloc[-1]
                delta = end_val - start_val
                growth = (delta / start_val) * 100 if start_val != 0 else 0
                
                key_metrics[metric] = {
                    "start": start_val,
                    "end": end_val,
                    "delta": delta,
                    "growth": growth
                }
            
            state['charts'] = charts
            state['key_metrics'] = key_metrics
        except Exception as e:
            state['error'] = f"Data processing error: {str(e)}"
        return state
    
    # Build workflow
    workflow = StateGraph(BusinessMetricsState)
    workflow.add_node("fetch_web_trends", fetch_web_trends)
    workflow.add_node("analyze_trends", analyze_trends)
    workflow.add_node("generate_charts", generate_charts_and_metrics)
    
    workflow.set_entry_point("fetch_web_trends")
    workflow.add_edge("fetch_web_trends", "analyze_trends")
    workflow.add_edge("analyze_trends", "generate_charts")
    workflow.add_edge("generate_charts", END)
    print("final stage")
    
    return workflow.compile()

# ======================== UI COMPONENTS ========================
def render_sidebar():
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    prev_model = st.session_state.model_name
    model_name = st.sidebar.selectbox(
        "AI Model", 
        [
            "gpt-4o", 
            "gpt-4-turbo", 
            "gpt-3.5-turbo", 
            "gemini-1.5-flash", 
            "deepseek-reasoner"
        ],
        index=0,
        key="model_selector",  # Important: use a dedicated key
        help="Select AI model to use for analysis"
    )

    if model_name != st.session_state.prev_model:
        st.session_state.prev_model = model_name
        st.session_state.model_name = model_name
        st.sidebar.success(f"Model changed to {model_name}")
        st.rerun()
    
    # Business Context Section
    st.sidebar.subheader("Business Context")
    
    # Context file upload
    context_file = st.sidebar.file_uploader(
        "Upload Business Document", 
        type=["txt", "pdf"],
        help="Upload TXT/PDF describing your business"
    )

    with st.sidebar.expander("📋 Sample Context", expanded=False):
        st.caption("Business context should include:")
        st.markdown("""
        - Core business model & industry  
        - Key products/services  
        - Target customers  
        - Competitive advantages  
        - Business goals  
        - Locations
        """, unsafe_allow_html=True)
        
        industry = st.selectbox("Quick Samples", 
                               ["Eco-Fashion", "SaaS", "Coffee", "Restaurant"],
                               index=0)
        
        if industry == "Eco-Fashion":
            st.code("""GreenThread Apparel
        Industry: Sustainable Fashion
        Products: Organic cotton apparel
        Target: Eco-conscious millennials
        Goals: 30% YOY growth, EU expansion
        Advantage: Carbon-neutral shipping
        Location: USA, GERMANY""", language="text")
        
        elif industry == "SaaS":
            st.code("""FlowStack Technologies
        Industry: B2B Software
        Product: AI productivity platform
        Target: Tech teams (50-500 employees)
        Pricing: $15/user/month
        Goal: 10K MAU by EOY
        Location: Europe, USA""", language="text")
            
        elif industry == "Coffee":
            st.code("""Mountain Peak Coffee
        Industry: Specialty Coffee
        Products: Single-origin beans
        Target: Urban professionals
        Advantage: Direct trade farmers
        Goal: 40% subscription growth
        Location: UK, Germany, USA""", language="text")
            
        elif industry == "Restaurant":
            st.code("""Urban Bistro
        Industry: Casual Dining
        Cuisine: Farm-to-table
        Target: Foodies (25-45)
        Advantage: Locally sourced ingredients
        Goal: Expand catering by 25%
        Location: India, USA, UK""", language="text")
    
    if context_file:
        try:
            extracted_text = extract_text_from_file(context_file)
            if len(extracted_text) > 200:
                st.session_state.context_text = extracted_text
                st.session_state.context_file = context_file
                st.sidebar.success("✅ Document processed! Click 'Set Context' to index it")
            else:
                st.sidebar.warning("Document too short (min 200 characters)")
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")
    
    # Manual context input
    manual_context = st.sidebar.text_area(
        "Or describe your business:",
        value=st.session_state.context_text,
        height=150,
        placeholder="We sell eco-friendly products...",
        help="Minimum 200 characters"
    )
    
    # Update context text if user types in the text area
    if manual_context != st.session_state.context_text:
        st.session_state.context_text = manual_context
        st.session_state.context_file = None
    
    # Show context status
    if st.session_state.context_set:
        st.sidebar.success("✅ Business context is set and indexed")
    else:
        st.sidebar.warning("ℹ️ Business context not set")
    
    # Set context button
    if st.sidebar.button("💼 Set Context", key="set_context", use_container_width=True):
        if len(st.session_state.context_text) >= 200:
            try:
                # Create Pinecone index
                with st.spinner("Indexing business context..."):
                    vector_store, namespace = create_pinecone_index(st.session_state.context_text)
                    st.session_state.pinecone_index = vector_store
                    st.session_state.context_namespace = namespace
                    st.session_state.business_context = st.session_state.context_text
                    st.session_state.context_set = True
                    st.session_state.context_set_time = datetime.now()
                st.sidebar.success("✅ Business context set and indexed!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error indexing context: {str(e)}")
        else:
            st.sidebar.error("Please provide at least 200 characters")
    
    # Clear context button
    if st.sidebar.button("🔄 Clear Context", use_container_width=True):
        st.session_state.business_context = ""
        st.session_state.context_text = ""
        st.session_state.context_file = None
        st.session_state.pinecone_index = None
        st.session_state.context_namespace = None
        st.session_state.context_set = False
        st.session_state.context_set_time = None
        st.sidebar.info("Context cleared")
    
    st.sidebar.markdown("---")
    
    # Business Metrics Section
    st.sidebar.subheader("Business Metrics")
    data_file = st.sidebar.file_uploader(
        "Upload Metrics CSV", 
        type=["csv"],
        help="Should include 'Date' column and metrics"
    )

    st.sidebar.markdown("### Get Sample Data")
    sample_industry = st.sidebar.selectbox(
        "Select industry template",
        ["E-Commerce", "SaaS", "Restaurant", "Retail"],
        index=0,
        key="sample_industry"
    )
    
    # Create sample data based on selection
    if sample_industry == "E-Commerce":
        sample_data = pd.DataFrame({
            "Date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "Revenue": [45000, 52000, 61000, 58500],
            "Website Traffic": [120000, 145000, 162000, 158000],
            "Conversion Rate": ["2.1%", "2.3%", "2.5%", "2.4%"],
            "Return Rate": ["4.2%", "3.8%", "3.5%", "4.0%"]
        })
    elif sample_industry == "SaaS":
        sample_data = pd.DataFrame({
            "Date": ["2024-01-08", "2024-01-15", "2024-01-22", "2024-01-29"],
            "Active Users": [1250, 1380, 1470, 1620],
            "Paid Conversions": [35, 42, 38, 45],
            "Churn Rate": ["1.2%", "1.1%", "1.0%", "0.9%"],
            "ARR": [185000, 192500, 201000, 218000]
        })
    elif sample_industry == "Restaurant":
        sample_data = pd.DataFrame({
            "Date": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04"],
            "Foot Traffic": [320, 290, 410, 480],
            "Avg Ticket Size": [24.50, 26.10, 22.80, 27.40],
            "Food Cost %": ["28%", "30%", "26%", "27%"],
            "Online Orders": [85, 92, 103, 121]
        })
    else:  # Retail
        sample_data = pd.DataFrame({
            "Date": ["2024-06-01", "2024-06-08", "2024-06-15", "2024-06-22"],
            "Sales": [12500, 14200, 13850, 16100],
            "Foot Traffic": [420, 380, 410, 510],
            "Avg Basket Size": [45.20, 48.10, 42.50, 46.80],
            "Conversion Rate": ["22.5%", "24.1%", "21.8%", "23.7%"]
        })
    
    # Convert to CSV for download
    csv = sample_data.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        "⬇️ Download Sample CSV",
        data=csv,
        file_name=f"{sample_industry}_sample_data.csv",
        mime="text/csv",
        use_container_width=True,
        help=f"Download {sample_industry} sample data template"
    )
    
    # Existing sample CSV section remains as reference
    with st.sidebar.expander("📈 Sample Formats", expanded=False):
        st.caption("CSV Requirements:")
        st.markdown("""
        - `Date` column (YYYY-MM-DD)  
        - At least 2 numeric metrics  
        - Consistent time intervals  
        """, unsafe_allow_html=True)
        
        st.markdown("<small>E-Commerce (Monthly)</small>", unsafe_allow_html=True)
        st.code("""Date,Revenue,Traffic
2024-01-01,45000,120000
2024-02-01,52000,145000""", language="csv")
        
        st.markdown("<small>SaaS (Weekly)</small>", unsafe_allow_html=True)
        st.code("""Date,Users,Churn
2024-01-08,1250,1.2%
2024-01-15,1380,1.1%""", language="csv")
    
    if data_file:
        try:
            st.session_state.business_data = pd.read_csv(data_file)
            st.sidebar.success("✅ Data uploaded!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

    if st.sidebar.button("🗑️ Clear Data", use_container_width=True):
        st.session_state.business_data = None
        st.sidebar.info("Data cleared")

    st.sidebar.markdown("---")
    
    # Agent documentation
    st.sidebar.subheader("Agent Capabilities")
    st.sidebar.markdown("""
    - **🔍 Supplier Agent**: 
      - Global supplier research
      - Vendor vetting & qualification
    
    - **💰 Pricing Agent**: 
      - Competitive analysis
      - Pricing strategy development
    
    - **📈 Trend Agent**: 
      - Market forecasting
      - Emerging trend identification
                        
    - **🏆 Competitor Agent**:  
      - Competitive benchmarking
      - Whitespace opportunity identification
                        
    - **✉️ Newsletter Agent**: 
      - Marketing email creation
      - Campaign copy generation
    
    - **📊 Metrics Agent**: 
      - Performance analysis
      - KPI tracking & optimization
    """)

def render_supplier_form():
    """Form for collecting supplier requirements"""
    with st.expander("🔍 Supplier Requirements", expanded=True):
        with st.form("supplier_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.query = st.text_input(
                    "Product/Service*", 
                    st.session_state.get("query", ""),
                    placeholder="Wind turbine parts"
                )
                
                st.session_state.specifications = st.text_area(
                    "Specifications*", 
                    st.session_state.get("specifications", ""),
                    placeholder="Technical requirements, certifications, etc.",
                    height=100
                )
                
            with col2:
                st.session_state.locations = st.multiselect(
                    "Preferred Locations",
                    ["USA", "Canada", "UK", "Germany", "France", "China", "Japan", "India"],
                    default=st.session_state.get("locations", [])
                )
                
                st.session_state.notes = st.text_area(
                    "Additional Notes", 
                    st.session_state.get("notes", ""),
                    placeholder="Urgency, budget constraints, etc.",
                    height=80
                )
            
            if st.form_submit_button("🚀 Find Suppliers", use_container_width=True):
                if st.session_state.query and st.session_state.specifications:
                    return True
                else:
                    st.error("Please fill in required fields (marked with *)")
    return False

def render_metrics_selection():
    """Render metric selection for business data"""
    if st.session_state.business_data is None:
        print("No business data available")
        return
    
    # Get available metrics
    numeric_cols = st.session_state.business_data.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric metrics found in the data")
        return

def render_business_metrics_results(state, unique_id):
    """Render results from business metrics agent with unique keys"""
    if not state:
        return
        
    # Display error if exists
    if state.get('error'):
        st.error(f"❌ Error: {state['error']}")
        return
        
    # Display charts - only these need keys
    st.subheader("📈 Historical Trend Charts")
    if state.get('charts'):
        for i, (metric, fig) in enumerate(state['charts'].items()):
            # Create unique key using metric name and index
            chart_key = f"chart_{metric}_{i}_{unique_id}"
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
    else:
        st.warning("No charts generated")
    
    # Display key metrics
    st.subheader("📌 Key Metrics Summary")
    if state.get('key_metrics'):
        for metric, values in state['key_metrics'].items():
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Start {metric}", f"{values['start']:,.2f}")
            col2.metric(f"End {metric}", f"{values['end']:,.2f}", f"{values['delta']:,.2f}")
            col3.metric(f"Growth %", f"{values['growth']:.1f}%")
    else:
        st.warning("No key metrics calculated")
    
    # Show insights
    st.subheader("🧠 AI-Generated Insights & Recommendations")
    if state.get('insights'):
        st.markdown(state['insights'])
    else:
        st.warning("No insights generated")
    
    # Show web results - st.code doesn't need a key
    st.subheader("🌐 Latest Web Trends")
    if state.get('web'):
        st.code(state['web'], language="markdown")
    else:
        st.warning("No web trends found")

def render_chat():
    """Main chat interface"""
    st.subheader("Business building assistant")
    st.markdown("Ask questions about your business, suppliers, pricing, trends, or metrics.")
    # Display messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Display business metrics results
            if msg.get("agent_type") == "business_metrics" and msg.get("data"):
                render_business_metrics_results(msg["data"], i)
            
            # Display supplier results
            if "suppliers" in msg:
                st.subheader(f"🔍 Found {len(msg['suppliers'])} Suppliers")
                for j, supplier in enumerate(msg["suppliers"]):
                    with st.expander(f"{j+1}. {supplier['name']}"):
                        st.markdown(f"**Website**: [{supplier['website']}]({supplier['website']})")
                        st.caption(supplier["description"])
    
    # User input
    if prompt := st.chat_input("Ask about your business..."):
        # Check if business context is set
        if not st.session_state.business_context:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🚫 **Action Required**\n\nPlease set your business context in the sidebar to enable AI assistance."
            })
            st.rerun()
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.query = prompt
        
        # Initialize supervisor state
        supervisor_state = {
            "query": prompt,
            "context": st.session_state.business_context,
            "agent_type": "unknown",
            "specifications": st.session_state.get("specifications", ""),
            "locations": st.session_state.get("locations", []),
            "exclude": [],
            "notes": st.session_state.get("notes", ""),
            "results": None,
            "feedback": "",
            "needs_data": False
        }
        
        # Run supervisor workflow
        supervisor_workflow = create_supervisor_workflow()
        result = supervisor_workflow.invoke(supervisor_state)
        st.session_state.supervisor_state = result
        
        # Handle agent routing
        if "supplier" in result["agent_type"]:
            if result.get("needs_data"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "🔍 Please Scroll to the top of the page and provide specifications and locations for supplier search"
                })
                st.session_state.show_supplier_form = True
            else:
                st.rerun()  # Will trigger supplier form render
        elif "business_metrics" in result["agent_type"]:
            if result.get("needs_data"):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "📊 Please upload your business metrics CSV using the sidebar"
                })
            else:
                with st.spinner("Analyzing business metrics..."):
                    # Prepare data for business metrics agent
                    business_data = st.session_state.business_data
                    
                    # Validate data before processing
                    if business_data is None or business_data.empty:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "❌ No business data available",
                            "agent_type": "business_metrics"
                        })
                        return
                    
                    if "Date" not in business_data.columns:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "❌ Data must contain a 'Date' column",
                            "agent_type": "business_metrics"
                        })
                        return
                    
                    try:
                        business_data["Date"] = pd.to_datetime(business_data["Date"], errors="coerce")
                        business_data = business_data.dropna(subset=["Date"])
                        business_data.sort_values("Date", inplace=True)
                        historic_json = business_data.to_json(orient="records", date_format="iso")
                        
                        # Run business metrics workflow
                        metrics_workflow = create_business_metrics_workflow()
                        metrics_state = {
                            "topic": prompt,
                            "historic": historic_json,
                            "web": "",
                            "insights": "",
                            "charts": {},
                            "key_metrics": {},
                            "error": None
                        }
                        result = metrics_workflow.invoke(metrics_state)
                        
                        # Add response to chat
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"## 📊 Business Metrics Analysis for '{prompt}'",
                            "agent_type": "business_metrics",
                            "data": result
                        })
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"❌ Error processing data: {str(e)}",
                            "agent_type": "business_metrics"
                        })
        else:
            # Handle other agents
            with st.spinner("Analyzing..."):
                handle_agent_response(prompt, result["agent_type"])
        st.rerun()
# ======================== AGENT RESPONSE HANDLER ========================

def handle_agent_response(prompt, agent_type):
    """Process user query with appropriate agent"""
    # Check business context exists (should already be checked, but adding for safety)
    if not st.session_state.business_context:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Business context not set - please configure in sidebar"
        })
        return
        
    # Route to appropriate agent
    if agent_type == "pricing":
        agent = PricingAgent(st.session_state.model_name)
        response = agent.analyze_pricing(prompt, st.session_state.business_context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response}"
        })
        
    elif agent_type == "trend":
        agent = TrendAgent(st.session_state.model_name)
        response = agent.analyze_trends(prompt, st.session_state.business_context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response}"
        })

    elif agent_type == "competitor":
        agent = CompetitorAgent(st.session_state.model_name)
        response = agent.analyze_competitors(prompt, st.session_state.business_context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response}"
        })

    elif agent_type == "newsletter":
        agent = NewsletterAgent(st.session_state.model_name)
        response = agent.generate_copy(prompt, st.session_state.business_context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{response}"
        })
        
    elif agent_type == "business_metrics":
        try:
            # Prepare and validate data
            business_data = st.session_state.business_data
            if business_data is None or business_data.empty:
                st.error("❌ No business data available")
                return
                
            if "Date" not in business_data.columns:
                st.error("❌ Data must contain a 'Date' column")
                return
            
            # Convert date and sort
            business_data["Date"] = pd.to_datetime(business_data["Date"], errors="coerce")
            business_data = business_data.dropna(subset=["Date"])
            business_data.sort_values("Date", inplace=True)
            historic_json = business_data.to_json(orient="records", date_format="iso")
            
            # Run business metrics workflow
            metrics_workflow = create_business_metrics_workflow()
            metrics_state = {
                "topic": prompt,
                "historic": historic_json,
                "web": "",
                "insights": "",
                "charts": {},
                "key_metrics": {},
                "error": None
            }
            result = metrics_workflow.invoke(metrics_state)
            
            # Add response to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"## 📊 Business Metrics Analysis for '{prompt}'",
                "agent_type": "business_metrics",
                "data": result
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Error processing data: {str(e)}",
                "agent_type": "business_metrics"
            })
        
    else:
        # Fallback to general agent
        agent = GeneralAgent(st.session_state.model_name)
        response = agent.answer_question(prompt, st.session_state.business_context)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

def handle_supplier_request():
    """Process supplier search request"""
    if not st.session_state.business_context:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ Business context not set - please configure in sidebar"
        })
        return
    with st.spinner("Searching global suppliers..."):
        agent = SupplierAgent(st.session_state.model_name)
        suppliers = agent.search_suppliers(
            st.session_state.query,
            st.session_state.specifications,
            st.session_state.locations,
            st.session_state.exclude,
            st.session_state.notes,
            st.session_state.business_context
        )
        
        if suppliers:
            content = f"## 🔍 Supplier Research: {st.session_state.query}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": content,
                "suppliers": suppliers
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ No suppliers found. Try broadening your search criteria."
            })

# ======================== MAIN APPLICATION ========================
def main():
    st.set_page_config(
        page_title="Business Pal", 
        page_icon="", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    st.title("Business Pal")
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Show context status
        if st.session_state.get("context_set"):
            st.success(f"✅ Business context is set: {st.session_state.business_context[:300]}...")
        else:
            st.error("🚨 **Action Required** - Please set your business context in the sidebar to enable AI assistance")
            
        model_name = st.session_state.get("model_name")
        if model_name == "gemini-1.5-flash":
            st.info("ℹ️ Using Google Gemini Pro model")
        elif model_name == "deepseek-reasoner":
            st.info("ℹ️ Using DeepSeek Chat model")
        else:
            st.info(f"ℹ️ Using OpenAI {model_name} model")
        # Metrics selection if data is available
        if st.session_state.business_data is not None:
            render_metrics_selection()
            
        # Show supplier form if requested
        if st.session_state.show_supplier_form:
            if render_supplier_form():
                if not st.session_state.business_context:
                    st.error("❌ Please set business context before searching for suppliers")
                else:
                    handle_supplier_request()
                    st.session_state.show_supplier_form = False
                    st.rerun()
        
        # Show chat interface
        render_chat()
    
    with col2:
        render_sidebar()

if __name__ == "__main__":
    main()
