from liquid import Template

general_prompt = {
    "system": Template("""
        You are an experienced medical expert, proficient in field {{field}}. Now, please answer the following QA questions.
    """),
    "user": Template("""
        The questions is: {{question}} 
        The options are: {{options}} 
        Please analyze this question carefully and provide the following two objects:
        (1) your only answer option
        (2) your confidence in each option. Note: The confidence indicates how likely you think the option is true. Your confidence level, please only include the numerical number in the range of 0-1. The sum of the confidence values for all options should equal 1.   
        Make sure the output is in JSON format like this: { "Answer": "C", "conf_each_option": { "A": 0.02, "B": 0.03, "C": 0.85, "D": 0.03, "E": 0.07 } }
    """),
}

assist_debate_prompt = {
    'system': Template("""
        Please act as a meticulous and objective analysis expert. Your task is to conduct an in-depth evaluation of a multiple-choice question and determine the best answer. Your analysis should be rigorous, evidence-based, and strictly confined to the information presented.
    """),
    'user': Template("""
        **Question:**
        {{question}}

        **Options:**
        {{options}}

        **Retrieved Knowledge Analysis:**
        {{rag_analysis}}

        **Analysis Standards:**
        1. **Accuracy**: Is the option factually correct and scientifically sound?
        2. **Completeness**: Does the option fully address what the question is asking?
        3. **Relevance**: Is the option directly related to the question's core focus?
        4. **Precision**: Is the option specific enough and not overly vague or general?
        5. **Context Appropriateness**: Does the option fit the context and scope of the question?
        6. **Logical Consistency**: Is the option internally consistent and free of contradictions?
        7. **Best Fit**: Among correct options, which one best answers the question?
        8. **Knowledge Support**: How well does each option align with the retrieved knowledge analysis?

       **Instructions:**
        Apply the above standards to evaluate each option systematically, incorporating insights from the retrieved knowledge analysis. The retrieved analysis provides additional context and expert knowledge that should inform your evaluation, but you should also apply critical thinking to assess its relevance and accuracy.

        Provide your analysis in the following JSON format:

        { "reasoning": ["Analysis step 1 (incorporating retrieved knowledge where relevant)", "Analysis step 2 (evaluating options against standards)", "Analysis step 3 (final decision rationale)"], "answer": "A/B/C/D" }
    """)
}

refined_debate = {
    'system': Template("""You are an expert in logical reasoning, critical thinking, and decision analysis. Your task is to analyze a given question along with several candidate answers. Each candidate answer is accompanied by a preliminary analysis and additional detailed reasoning."""),
    'user': Template("""                     
        **Question:**
        {{question}}

        **Initial Judgement:**
        {{initial_judgement}}
        
        ** Assist Information**
        {{assist_infos}}

        **Instructions:**
        Carefully review the question, initial answer, and all supporting analyses. Consider the following factors:

        1. **Consensus Among Analyses**: Do most analyses point to the same answer?
        2. **Quality of Reasoning**: Which analyses provide the most sound and logical reasoning?
        3. **Evidence Strength**: Which option has the strongest supporting evidence across analyses?
        4. **Initial Answer Validation**: Does the initial answer align with the analytical evidence?
        5. **Contradictions Resolution**: How do you resolve any conflicting viewpoints in the analyses?
        6. **Overall Coherence**: Which answer creates the most coherent and complete solution?

        **Output Format:**
        Provide your final decision in the following JSON format:
        
        ```json
        {
            "best_answer": "A or B or C or D",
            "confidence_level": "high/medium/low", 
            "reasoning": "detailed explanation of why this answer is the best by fully considering the given analyses"
        }

        Please ensure your analysis is objective, detailed, and based on evidence and logic.
        
    """)
}

rag_get_question_whole_analysis_prompt = {
    "system": Template("You are {{setup}}."),
    "user": Template("""You are {{setup}}. Your task is to help analyze a multiple-choice question comprehensively.
        Here is the question: "{{main_question}}"
        Options: {{options}}
        Please generate three distinct and critical questions that would help provide a thorough analysis of this question from your perspective as {{setup}}.
        The questions should be probing and aim to uncover key information related to the main question and all options.
        Return your response as a JSON object with a single key "questions" which is a list of the three generated questions.
        The questions are then fed into a retrieval system to retrieve relevant information, so they should be formatted as queries for an embedding model/BM25 index, like search engine queries.
        Example format: { "questions": ["question 1", "question 2", "question 3"] }
    """)
}

rag_whole_analysis_prompt = {
    "system": Template("You are {{setup}}."),
    "user": Template("""Please analyze the following multiple-choice question based on the provided retrieved context.
        Original Question: "{{question}}"
        Options: {{options}}
        Retrieval Query: 
        ---
        "{{retrieve_query}}"
        ---
        Retrieved Context:
        ---
        {{formatted_context}}
        ---
        Your task is to provide a comprehensive analysis of the question and all options using the given context from your perspective as {{setup}}. Please be aware that the retrieved information (RAG) may be irrelevant or contain inaccuracies, so you must think critically and evaluate the information's reliability. 
        Provide your analysis focusing on:
        1. Understanding of the question
        2. Evaluation of each option
        3. Your reasoning process
        Output your analysis as structured paragraphs.
    """)
}