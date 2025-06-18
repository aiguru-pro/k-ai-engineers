# Prompt Engineering Homework Assignments
## Kognitic Clinical Intelligence Workshop

---

## 🎯 Beginner Level Homework (Week 1)

### Assignment 1: Prompt Improvement Challenge
**Task:** Take these basic prompts and engineer them for better results

**Given Basic Prompts:**
1. "Summarize this research paper"
2. "Check this code for bugs" 
3. "Write a product description"

**Requirements:**
- Add specific role definitions
- Include output format requirements
- Add quality standards
- Test with real examples and compare results

**Deliverable:** Before/after comparison document showing improved prompts and their outputs

---

### Assignment 2: Clinical Domain Adaptation
**Task:** Create domain-specific prompt templates for different clinical scenarios

**Scenarios to Cover:**
1. **Drug Safety Analysis** - Analyze adverse event reports
2. **Clinical Trial Recruitment** - Assess patient eligibility  
3. **Medical Literature Review** - Summarize key findings from research papers

**Requirements:**
- Each template must include: Role, Context, Task, Format, Quality Standards
- Include 3 example inputs and expected outputs
- Document what makes each domain unique

**Template Format:**
```
Role: You are a [specific clinical expert]
Context: [business objective and stakeholder needs]
Task: [specific analysis requirements]
Format: [structured output requirements]
Quality Standards: [validation and accuracy requirements]
```

---

## 🎯 Intermediate Level Homework (Week 2)

### Assignment 3: Chain-of-Thought Implementation
**Task:** Build a systematic analysis system for competitive intelligence

**Problem:** Analyze competitor drug launches and predict market impact

**Requirements:**
- Create a 5-step chain-of-thought process
- Include decision points and confidence scoring
- Handle different therapeutic areas (oncology, cardiology, neurology)
- Provide executive summary format

**Chain-of-Thought Framework:**
```
Step 1: Market Context Analysis
Step 2: Clinical Differentiation Assessment
Step 3: Competitive Landscape Evaluation
Step 4: Commercial Viability Analysis
Step 5: Strategic Impact Prediction
```

**Test Cases:** Students get 3 real drug approval announcements to analyze

---

### Assignment 4: Multi-Modal Clinical Analysis
**Task:** Create prompts that integrate different data types

**Challenge:** Analyze a clinical trial that includes:
- Trial protocol (text)
- Enrollment data (numbers/charts)
- Competitive landscape (market data)
- Regulatory guidance (policy documents)

**Requirements:**
- Design prompts that can synthesize across data types
- Create structured output for different stakeholders (medical, commercial, regulatory)
- Include validation steps for data accuracy

**Output Formats Required:**
- **Medical Affairs:** Scientific summary with clinical implications
- **Commercial Team:** Market opportunity and competitive positioning
- **Regulatory:** Approval pathway and timeline assessment

---

## 🎯 Advanced Level Homework (Week 3)

### Assignment 5: Dynamic Prompt System
**Task:** Build adaptive prompts based on user input and context

**Challenge:** Create a system that:
- Classifies incoming queries automatically
- Selects appropriate analysis framework
- Adjusts detail level based on user expertise
- Provides different outputs for different roles

**Example Use Case:** Kognitic platform that serves both junior analysts and C-suite executives

**Implementation Requirements:**
```python
def classify_query(user_input, user_role):
    # Determine analysis type needed
    
def select_prompt_template(query_type, user_expertise):
    # Choose appropriate prompt framework
    
def generate_response(template, data, output_level):
    # Create role-appropriate analysis
```

---

### Assignment 6: Quality Control Framework
**Task:** Design validation and error detection for AI outputs

**Requirements:**
- Create prompts that can verify their own outputs
- Build fact-checking workflows for clinical claims
- Design confidence scoring systems
- Handle edge cases and data inconsistencies

**Quality Control Components:**
1. **Fact Verification:** Cross-reference clinical claims
2. **Consistency Checking:** Ensure logical coherence
3. **Completeness Assessment:** Verify all requirements met
4. **Confidence Scoring:** Rate reliability of conclusions

**Deliverable:** Complete QA system with test cases

---

## 🎯 Real-World Application Projects (Week 4)

### Assignment 7: Mini-Application Development
**Task:** Build a complete application using advanced prompt engineering

**Project Options:**

#### Option A: Biomarker Strategy Analyzer
- **Input:** Trial design and biomarker approach
- **Output:** Commercial viability assessment
- **Features:** Population sizing, competitive analysis, regulatory pathway

#### Option B: Regulatory Pathway Predictor
- **Input:** Drug profile and indication
- **Output:** Approval strategy recommendation
- **Features:** Timeline prediction, risk assessment, precedent analysis

#### Option C: Clinical Trial Timeline Forecaster
- **Input:** Trial parameters and competitive landscape
- **Output:** Completion date prediction
- **Features:** Enrollment modeling, risk factors, milestone tracking

**Requirements:**
- Professional user interface (Gradio/Streamlit)
- Multiple prompt templates working together
- Error handling and edge case management
- Documentation and user guide

**Application Structure:**
```
├── prompt_templates/
│   ├── competitive_analysis.py
│   ├── timeline_prediction.py
│   └── regulatory_assessment.py
├── data_processing/
│   ├── trial_database.py
│   └── validation.py
├── interface/
│   └── gradio_app.py
├── tests/
│   └── test_prompts.py
└── documentation/
    ├── user_guide.md
    └── technical_docs.md
```

---

## 🎯 Ongoing Practice Assignments

### Weekly Challenge: Prompt Debugging
**Format:** Students get "broken" prompts that produce poor results

**Common Issues to Debug:**
- **Hallucination:** Prompt that creates fake clinical data
- **Format Issues:** Inconsistent or unusable output structure
- **Context Problems:** Missing critical business context
- **Edge Cases:** Prompt fails with unusual inputs
- **Scope Creep:** Prompt tries to do too many things

**Example Broken Prompt:**
```
"Analyze this drug and tell me everything about it and what will happen in the market and if it's good or bad and what companies should do about it."
```

**Task:** Diagnose issues and fix the prompts

---

### Peer Review Exercise
**Format:** Students exchange their prompt templates for review

**Review Criteria:**
1. **Clarity and Specificity** (25%)
   - Is the role clearly defined?
   - Are requirements specific and measurable?
   
2. **Business Relevance** (25%)
   - Does it address real clinical intelligence needs?
   - Is the output actionable for stakeholders?
   
3. **Technical Quality** (25%)
   - Proper use of prompt engineering techniques?
   - Appropriate output formatting?
   
4. **Edge Case Handling** (25%)
   - Does it handle unusual inputs gracefully?
   - Are error conditions managed?

**Peer Review Template:**
```markdown
## Prompt Review for [Student Name]

### Strengths:
- 

### Areas for Improvement:
- 

### Specific Suggestions:
- 

### Overall Score: [1-10]
```

---

## 🎯 Assessment Rubric

### Evaluation Criteria:

#### Technical Quality (40%)
- **Prompt Structure:** Clear role, context, task, format, standards
- **Advanced Techniques:** Effective use of CoT, few-shot, etc.
- **Error Handling:** Robust response to edge cases
- **Code Quality:** Clean, maintainable implementation

#### Business Relevance (30%)
- **Problem Alignment:** Addresses real clinical intelligence needs
- **Stakeholder Value:** Provides actionable insights for decision-makers
- **Industry Context:** Demonstrates understanding of pharma/biotech challenges
- **Output Quality:** Professional, executive-ready deliverables

#### Innovation (20%)
- **Creative Solutions:** Novel approaches to prompt engineering challenges
- **Integration:** Clever combination of multiple techniques
- **Efficiency:** Optimized for performance and cost
- **Scalability:** Designed for real-world application

#### Documentation (10%)
- **Clarity:** Clear explanations of design choices
- **Completeness:** Comprehensive usage instructions
- **Examples:** Well-crafted test cases and demonstrations
- **Maintenance:** Guidelines for updates and improvements

### Grading Scale:
- **A (90-100%):** Professional-quality work ready for production use
- **B (80-89%):** Solid implementation with minor improvements needed
- **C (70-79%):** Functional but requires significant refinement
- **D (60-69%):** Basic understanding demonstrated, major gaps remain
- **F (<60%):** Does not meet minimum requirements

---

## 🎯 Bonus Challenges

### Advanced Research Project
**Options:**
1. **Literature Review:** Research and implement cutting-edge prompt engineering techniques
2. **Industry Benchmark:** Compare your prompts against commercial clinical intelligence tools
3. **Optimization Study:** A/B test different prompt variations and measure performance
4. **Academic Paper:** Write a research paper on prompt engineering applications in pharma

### Open Source Contribution
**Opportunities:**
- Contribute prompt templates to a shared library
- Create educational resources for other students
- Build tools that help others engineer better prompts
- Maintain a public repository of clinical intelligence prompts

### Industry Partnership Project
**Real-World Application:**
- Partner with a biotech company on actual competitive intelligence needs
- Develop prompts for real clinical trial analysis
- Present findings to industry professionals
- Incorporate feedback for continuous improvement

---

## 📚 Suggested Learning Resources

### Daily Practice
1. **Keep a Prompt Journal** - Document what works and what doesn't
2. **15-Minute Daily Practice** - Continuous improvement routine
3. **Real Data Usage** - Practice with actual clinical trials and drug development scenarios
4. **Iterative Refinement** - Continuously improve based on results

### Community Engagement
1. **Join Communities** - Participate in prompt engineering forums
2. **Share Knowledge** - Contribute to discussions and help others
3. **Stay Current** - Follow latest developments in the field
4. **Network** - Connect with other practitioners

### Professional Development
1. **Industry Conferences** - Attend AI and pharma conferences
2. **Webinars** - Participate in relevant educational sessions
3. **Certifications** - Pursue relevant AI/ML certifications
4. **Mentorship** - Find mentors in the field

### Recommended Reading
1. **Research Papers** - Latest prompt engineering research
2. **Industry Reports** - AI applications in pharmaceutical industry
3. **Technical Blogs** - Practitioner insights and best practices
4. **Case Studies** - Real-world implementation examples

---

## 📋 Homework Submission Guidelines

### Submission Format:
- **Code:** GitHub repository with clear README
- **Documentation:** Markdown files with examples
- **Analysis:** PDF report with findings and insights

## 🎯 Success Tips

### For Better Results:
1. **Start Early** - Don't wait until the last minute
2. **Test Thoroughly** - Use multiple examples to validate prompts
3. **Get Feedback** - Ask peers and instructors for input
4. **Iterate Often** - Continuous improvement is key
5. **Document Everything** - Keep track of what works and why

### Common Pitfalls to Avoid:
1. **Over-Engineering** - Keep prompts as simple as possible while meeting requirements
2. **Ignoring Edge Cases** - Always test with unusual inputs
3. **Poor Documentation** - Make your work understandable to others
4. **Neglecting Business Context** - Remember the real-world application
5. **Skipping Validation** - Always verify your outputs make sense

Remember: The goal is not just to complete assignments, but to develop skills that will make you more effective in your professional work with clinical intelligence and AI systems.
