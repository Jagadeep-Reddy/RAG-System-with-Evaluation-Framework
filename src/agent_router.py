import json
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
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
        self.llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
        
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
            sub_queries = json.loads(result)
            if not isinstance(sub_queries, list):
                return [query]
            return sub_queries
        except Exception as e:
            print(f"Decomposition parsing failed: {e}")
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
            
        # Execute sub-queries in parallel
        with ThreadPoolExecutor() as executor:
            sub_answers = list(executor.map(self._process_sub_query, sub_queries))
            
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
