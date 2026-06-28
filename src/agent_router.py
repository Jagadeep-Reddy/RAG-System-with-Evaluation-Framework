import json
from concurrent.futures import ThreadPoolExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.retrieval import HybridRetriever
from src.generation import RAGGenerator

class AgenticRouter:
    """
    Part D: Agentic RAG implementation handling query decomposition 
    and multi-hop reasoning for complex financial queries.
    """
    def __init__(self, retriever: HybridRetriever, generator: RAGGenerator):
        self.retriever = retriever
        self.generator = generator
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
        
        self.decomposer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query decomposition agent. 
            If a query is complex, comparative, or requires multi-hop reasoning across 
            different entities or time periods, break it down into an array of sub-queries.
            Return ONLY a JSON list of strings representing the sub-queries.
            If the query is simple, return a JSON list with just the original query.
            Example input: "Compare Apple and Microsoft R&D spending in 2023."
            Example output: ["What was Apple's R&D spending in 2023?", "What was Microsoft's R&D spending in 2023?"]"""),
            ("user", "{question}")
        ])
        
        self.synthesizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a synthesis agent. 
            You must combine the answers to sub-queries to form a comprehensive answer 
            to the user's original complex query.
            Ensure you maintain all citations [Document: X, Page: Y] provided in the sub-answers.
            
            Original Query: {original_query}
            
            Sub-Query Answers:
            {sub_answers}
            """),
            ("user", "Synthesize the final comparative answer.")
        ])

    def decompose(self, query: str) -> list[str]:
        """Breaks down complex query into sub-queries."""
        chain = self.decomposer_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"question": query})
        try:
            cleaned_result = result.strip()
            if cleaned_result.startswith("```"):
                # Filter out lines starting with markdown block delimiters (e.g. ```json or ```)
                lines = [line for line in cleaned_result.splitlines() if not line.strip().startswith("```")]
                cleaned_result = "\n".join(lines).strip()
                
            sub_queries = json.loads(cleaned_result)
            if not isinstance(sub_queries, list):
                return [query]
            return sub_queries
        except Exception as e:
            print(f"Decomposition parsing failed: {e}. Raw result: {result}")
            return [query]

    def _process_sub_query(self, sub_query: str) -> str:
        """Retrieves and generates answer for a single sub-query."""
        docs = self.retriever.retrieve(sub_query, top_n=5)
        return self.generator.generate(sub_query, docs)

    def route_and_execute(self, query: str) -> str:
        """
        Main entry point for agentic execution.
        Decomposes query -> Parallel retrieval/generation -> Synthesis
        """
        sub_queries = self.decompose(query)
        
        if len(sub_queries) == 1:
            print(f"Simple query detected. Executing standard RAG...")
            return self._process_sub_query(query)
            
        print(f"Complex query decomposed into {len(sub_queries)} sub-queries:")
        for i, sq in enumerate(sub_queries):
            print(f"  {i+1}. {sq}")
            
        # Execute sub-queries sequentially to avoid concurrent rate limit errors on the free-tier API
        import time
        sub_answers = []
        for sq in sub_queries:
            sub_answers.append(self._process_sub_query(sq))
            time.sleep(1.0) # Grace period to prevent triggering rapid RPM rate limits
            
        # Synthesize
        formatted_answers = ""
        for q, a in zip(sub_queries, sub_answers):
            formatted_answers += f"--- Sub-Query: {q} ---\n{a}\n\n"
            
        print("Synthesizing final multi-hop response...")
        synth_chain = self.synthesizer_prompt | self.llm | StrOutputParser()
        final_answer = synth_chain.invoke({
            "original_query": query,
            "sub_answers": formatted_answers
        })
        
        return final_answer
