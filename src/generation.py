from typing import List
from operator import itemgetter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class RAGGenerator:
    """
    Handles prompt engineering, strict citations, and hallucination detection.
    """
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
        
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous financial analyst. 
            You must answer the user's question using ONLY the provided context blocks below.
            
            CRITICAL INSTRUCTIONS:
            1. Every single factual claim you make MUST end with a bracketed citation pointing to its source.
            2. The citation format must be exactly: [Document: <source_doc>, Page: <page>].
            3. If the context does not contain enough information to answer the question, state: "Insufficient context to answer."
            4. Do not invent metrics or facts.
            5. Note that the documents provided (e.g., AAPL_10K.pdf, MSFT_10K.pdf) are the annual reports (10-K filings) for the fiscal year 2023. Therefore, unless another year is explicitly stated in the text, you can assume the reported figures in these documents are for the fiscal year 2023.
            
            Context Blocks:
            {context}"""),
            ("user", "Question: {question}")
        ])
        
    def _format_docs(self, docs: List[Document]) -> str:
        """Inject metadata into the prompt for the LLM to cite."""
        formatted_blocks = []
        for d in docs:
            doc_name = d.metadata.get('source_doc', 'Unknown')
            page_num = d.metadata.get('page', 'Unknown')
            formatted_blocks.append(
                f"--- SOURCE: doc={doc_name}, page={page_num} ---\n{d.page_content}\n"
            )
        return "\n\n".join(formatted_blocks)

    def generate(self, question: str, retrieved_docs: List[Document]) -> str:
        """
        Executes standard LCEL chain for generation using retrieved documents.
        """
        chain = (
            {"context": itemgetter("docs") | RunnableLambda(self._format_docs), 
             "question": itemgetter("question")}
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke({"docs": retrieved_docs, "question": question})

    def generate_with_self_consistency(self, question: str, retrieved_docs: List[Document]) -> str:
        """
        Part B3: Hallucination Detection via Self-Consistency.
        Runs the prompt 3 times at temp > 0. If answers drastically conflict, raise a warning.
        """
        # Create a more creative LLM for sampling
        sampling_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        chain = (
            {"context": itemgetter("docs") | RunnableLambda(self._format_docs), 
             "question": itemgetter("question")}
            | self.qa_prompt
            | sampling_llm
            | StrOutputParser()
        )
        
        print("Running self-consistency checks...")
        responses = chain.batch([{"docs": retrieved_docs, "question": question} for _ in range(3)])
        
        # Meta-prompt to vote/check conflicts
        vote_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI adjudicator. Below are 3 generated answers to the same question. Determine the consensus answer. If the answers fundamentally contradict each other regarding financial figures or facts, output 'WARNING: HALLUCINATION DETECTED' followed by a synthesized safe answer."),
            ("user", f"Question: {question}\n\nAnswer 1: {responses[0]}\n\nAnswer 2: {responses[1]}\n\nAnswer 3: {responses[2]}")
        ])
        
        voter_chain = vote_prompt | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) | StrOutputParser()
        final_answer = voter_chain.invoke({})
        return final_answer
