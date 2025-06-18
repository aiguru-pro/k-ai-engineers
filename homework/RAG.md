# 🧠 RAG Workshop - Homework Assignments
## Kognitic AI for Engineers - Practice Exercises

---

## 📚 **ASSIGNMENT 1: Basic RAG Extension (Beginner)**
**Estimated Time:** 2-3 hours  
**Learning Objectives:** Understanding chunking strategies, embedding models, and evaluation

### **Task:**
Extend the Basic RAG system with your own medical/scientific documents and compare different approaches.

### **Requirements:**

#### **Part A: Document Collection (30 minutes)**
1. **Find 3-5 medical research papers** from your field of interest
   - PubMed abstracts, FDA drug labels, clinical guidelines, etc.
   - Convert PDFs to text (copy-paste is fine for this exercise)
   - Each document should be 500-2000 words

#### **Part B: Chunking Strategy Comparison (45 minutes)**
2. **Implement 3 different chunking strategies:**
   ```python
   def chunk_by_sentences(text, sentences_per_chunk=3):
       # Your implementation here
       pass
   
   def chunk_by_characters(text, chunk_size=1000, overlap=100):
       # Your implementation here  
       pass
   
   def chunk_by_paragraphs(text):
       # Your implementation here
       pass
   ```

3. **Compare retrieval quality** for the same query across all chunking methods
   - Use 5 test questions about your documents
   - Record similarity scores for each method

#### **Part C: Embedding Model Comparison (45 minutes)**
4. **Test 3 different embedding models:**
   - `all-MiniLM-L6-v2` (baseline from workshop)
   - `all-mpnet-base-v2` (better quality)
   - `sentence-transformers/all-distilroberta-v1` (different architecture)

5. **Create a comparison table:**
   | Query | Model | Top Result | Similarity Score | Relevant? |
   |-------|-------|------------|------------------|-----------|
   | "What are the side effects?" | MiniLM | Chunk 3 | 0.82 | Yes |

#### **Part D: Evaluation & Analysis (30 minutes)**
6. **Write a brief analysis (300-500 words):**
   - Which chunking strategy worked best for your documents and why?
   - Which embedding model performed better for your domain?
   - What are the trade-offs between speed and quality?

### **Deliverables:**
- Jupyter notebook with your implementation
- Comparison tables/charts
- Brief written analysis
- Test it on at least 10 different queries

### **Bonus Points:**
- Add a simple evaluation metric (precision@3)
- Visualize embedding similarities using t-SNE or UMAP
- Test with non-English medical documents

---

## 🔍 **ASSIGNMENT 2: Advanced RAG for Your Domain (Intermediate)**
**Estimated Time:** 4-6 hours  
**Learning Objectives:** Hybrid search, metadata extraction, domain-specific optimization

### **Task:**
Build a domain-specific RAG system for your field (oncology, cardiology, pharma, biotech, etc.)

### **Requirements:**

#### **Part A: Domain-Specific Data Collection (1 hour)**
1. **Collect 10-15 documents** relevant to your specialization:
   - Clinical trial protocols, drug monographs, treatment guidelines
   - Research papers, regulatory documents, product inserts
   - Ensure variety in document types and complexity

#### **Part B: Enhanced Metadata Extraction (2 hours)**
2. **Create a metadata extraction system:**
   ```python
   def extract_medical_metadata(text, doc_type):
       metadata = {
           'document_type': doc_type,
           'therapeutic_area': None,  # Extract from text
           'drug_class': None,        # Identify drug mentions
           'study_phase': None,       # For clinical trials
           'patient_population': None, # Demographics/conditions
           'endpoints': [],           # Primary/secondary endpoints
           'safety_signals': [],      # Adverse events mentioned
           'publication_year': None,  # Extract if available
           'evidence_level': None     # Quality of evidence
       }
       # Your implementation here
       return metadata
   ```

3. **Implement domain-specific boosting:**
   - Boost recent documents (publication year)
   - Boost high-evidence documents (systematic reviews > case studies)
   - Boost documents matching therapeutic area of query

#### **Part C: Advanced Query Processing (1.5 hours)**
4. **Add query enhancement:**
   ```python
   def enhance_medical_query(query):
       # Add medical synonyms
       # Expand abbreviations (MI -> myocardial infarction)
       # Add related terms
       return enhanced_query
   
   def classify_query_intent(query):
       # Classify as: safety, efficacy, dosing, contraindications, etc.
       return intent_type
   ```

#### **Part D: Custom Evaluation (1 hour)**
5. **Create domain-specific evaluation:**
   - Generate 20 realistic questions from your field
   - Include different query types (factual, comparative, safety-focused)
   - Evaluate using medical accuracy, not just relevance

#### **Part E: Results Analysis (30 minutes)**
6. **Analyze performance:**
   - Which types of queries work best/worst?
   - How does metadata boosting affect results?
   - What are the main failure modes?

### **Deliverables:**
- Complete RAG system with domain optimization
- 20 test queries with expected answers
- Performance analysis report
- Demo video (5 minutes) showing your system

### **Bonus Points:**
- Add citation tracking (which specific sections answer which parts)
- Implement contradiction detection between documents
- Add a simple web interface using Streamlit

---

## 🚀 **ASSIGNMENT 3: Production-Ready RAG System (Advanced)**
**Estimated Time:** 8-12 hours  
**Learning Objectives:** Scalability, monitoring, evaluation metrics, deployment

### **Task:**
Build a production-ready RAG system that could be deployed in a real clinical/research environment.

### **Requirements:**

#### **Part A: Scalable Data Pipeline (3 hours)**
1. **Build an automated ingestion pipeline:**
   ```python
   class DocumentProcessor:
       def __init__(self):
           self.supported_formats = ['.pdf', '.txt', '.docx']
           self.processing_queue = []
       
       def process_batch(self, file_paths):
           # Batch processing with progress tracking
           pass
       
       def extract_text_from_pdf(self, pdf_path):
           # Handle various PDF types
           pass
       
       def detect_document_type(self, text):
           # Auto-classify document types
           pass
   ```

2. **Implement incremental updates:**
   - Track document versions and changes
   - Update only modified documents
   - Handle document deletions

#### **Part B: Advanced Search & Reranking (3 hours)**
3. **Implement sophisticated retrieval:**
   ```python
   class AdvancedRetriever:
       def __init__(self):
           self.semantic_index = None  # FAISS/ChromaDB
           self.keyword_index = None   # BM25/Elasticsearch
           self.graph_index = None     # Knowledge graph (optional)
       
       def hybrid_search(self, query, filters=None):
           # Combine multiple search strategies
           pass
       
       def rerank_with_cross_encoder(self, query, candidates):
           # Use cross-encoder for reranking
           pass
   ```

4. **Add query routing:**
   - Route different query types to different strategies
   - Simple factual → keyword search
   - Complex reasoning → semantic search + reranking

#### **Part C: Comprehensive Evaluation (2 hours)**
5. **Build evaluation framework:**
   ```python
   class RAGEvaluator:
       def __init__(self):
           self.metrics = {
               'retrieval': ['precision@k', 'recall@k', 'mrr'],
               'generation': ['faithfulness', 'answer_relevancy', 'context_relevancy'],
               'domain_specific': ['medical_accuracy', 'safety_completeness']
           }
       
       def evaluate_retrieval(self, queries, ground_truth):
           pass
       
       def evaluate_generation(self, qa_pairs):
           pass
       
       def generate_evaluation_report(self):
           pass
   ```

#### **Part D: Monitoring & Logging (2 hours)**
6. **Implement production monitoring:**
   ```python
   class RAGMonitor:
       def __init__(self):
           self.metrics_collector = MetricsCollector()
           self.logger = logging.getLogger('rag_system')
       
       def log_query(self, query, results, user_feedback=None):
           pass
       
       def detect_performance_drift(self):
           pass
       
       def generate_usage_analytics(self):
           pass
   ```

#### **Part E: Deployment & API (2 hours)**
7. **Create production API:**
   ```python
   # FastAPI or Flask application
   @app.post("/search")
   async def search_documents(query: QueryRequest):
       # Rate limiting, authentication, caching
       pass
   
   @app.get("/health")
   async def health_check():
       pass
   ```

### **Deliverables:**
- Complete production-ready codebase
- Docker deployment configuration
- API documentation
- Performance benchmarking report
- Monitoring dashboard
- Deployment guide

### **Bonus Points:**
- Deploy to cloud (AWS/GCP/Azure)
- Add user authentication and access control
- Implement A/B testing for different retrieval strategies
- Add automated model retraining pipeline

---

## 🎯 **ASSIGNMENT 4: Research Project (Graduate Level)**
**Estimated Time:** 15-20 hours  
**Learning Objectives:** Original research, novel approaches, academic contribution

### **Task:**
Conduct original research on a specific RAG challenge relevant to biomedical applications.

### **Suggested Research Topics:**

#### **Option A: Biomedical Knowledge Graph RAG**
- Integrate structured knowledge (UMLS, DrugBank) with unstructured text
- Compare graph-enhanced RAG vs traditional RAG
- Focus on drug-drug interactions or disease relationships

#### **Option B: Multi-modal Medical RAG**
- Combine text with medical images, tables, or molecular structures
- Handle complex medical documents with mixed content types
- Evaluate on real clinical scenarios

#### **Option C: Federated RAG for Healthcare**
- RAG system that works across multiple institutions without sharing data
- Privacy-preserving retrieval and generation
- Compliance with HIPAA/GDPR requirements

#### **Option D: Temporal RAG for Clinical Guidelines**
- Handle evolving medical knowledge and guidelines
- Track changes in recommendations over time
- Version-aware retrieval and temporal reasoning

### **Research Requirements:**
1. **Literature Review** (20+ relevant papers)
2. **Novel Methodology** with clear contributions
3. **Experimental Validation** on realistic datasets
4. **Comparison** with existing baselines
5. **Statistical Analysis** of results
6. **Discussion** of limitations and future work

### **Deliverables:**
- Research paper (8-12 pages, conference format)
- Complete implementation with reproducible experiments
- Presentation (20 minutes + Q&A)
- Code repository with documentation
- Dataset (if creating new evaluation data)

---

## 💡 **Tips for Success**

### **For All Assignments:**
- Start early and ask questions
- Document your thought process and decisions
- Test thoroughly with diverse queries
- Consider real-world constraints and limitations

### **Technical Best Practices:**
- Use version control (Git) from the beginning
- Write unit tests for core functionality
- Profile your code for performance bottlenecks
- Handle edge cases and error conditions

### **Domain Knowledge:**
- Consult with medical professionals for realistic scenarios
- Use established medical terminologies and standards
- Consider regulatory and ethical implications
- Stay updated with latest RAG research and techniques

---

## 📚 **Additional Resources**

### **Datasets for Practice:**
- **PubMed Central:** Open access biomedical literature
- **ClinicalTrials.gov:** Clinical trial protocols and results
- **FDA Orange Book:** Drug approval information
- **WHO Drug Information:** International drug safety data
- **MIMIC-III:** ICU data (requires training/approval)

### **Evaluation Frameworks:**
- **RAGAS:** RAG evaluation framework
- **TruLens:** Evaluation and monitoring for LLM applications
- **LangSmith:** Debugging and testing LLM applications

### **Advanced Tools:**
- **ChromaDB/Pinecone:** Production vector databases
- **Haystack/LangChain:** RAG frameworks
- **Weights & Biases:** Experiment tracking
- **Gradio/Streamlit:** Quick UI development

### **Medical NLP Resources:**
- **scispaCy:** Scientific text processing
- **BioBERT/ClinicalBERT:** Domain-specific models
- **UMLS:** Unified Medical Language System
- **MetaMap:** Medical concept extraction
