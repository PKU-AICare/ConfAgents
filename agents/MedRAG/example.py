from src.medrag import MedRAG

question = "A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral"
options = {
    "A": "paralysis of the facial muscles.",
    "B": "paralysis of the facial muscles and loss of taste.",
    "C": "paralysis of the facial muscles, loss of taste and lacrimation.",
    "D": "paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation."
}
cot = MedRAG(llm_name="OpenAI/gpt-3.5-turbo-16k", rag=True)
answer, _, _ = cot.answer(question=question, options=options, corpus_name="pubmed",retriever="BM25")
print(f"Final answer in json with rationale: {answer}")
# {
#   "step_by_step_thinking": "Compression of the facial nerve at the stylomastoid foramen will affect the function of the facial nerve. The facial nerve is responsible for innervating the muscles of facial expression, including those involved in smiling, frowning, and closing the eyes. It also carries taste sensation from the anterior two-thirds of the tongue. Additionally, the facial nerve controls tear production (lacrimation) and salivation. Therefore, compression of the facial nerve at the stylomastoid foramen will cause paralysis of the facial muscles (A), loss of taste (B), lacrimation (C), and decreased salivation (D).", 
#   "answer_choice": "D"
# }