"""
RAG Enhancement Module
Implements advanced RAG features: reranking, multi-hop retrieval, query expansion
Based on latest Agno v2 documentation and best practices (2025)
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import os

logger = logging.getLogger(__name__)

# Try to import Cohere for reranking
try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    logger.warning("Cohere not available, reranking will be disabled")

# Try to import Agno knowledge components
try:
    from agno.knowledge import Knowledge
    from agno.vectordb.lancedb import LanceDb, SearchType
    from agno.knowledge.embedder.openai import OpenAIEmbedder
    AGNO_KNOWLEDGE_AVAILABLE = True
except ImportError:
    AGNO_KNOWLEDGE_AVAILABLE = False
    logger.warning("Agno knowledge components not available")


class Reranker:
    """
    Reranking component using Cohere API
    Improves retrieval relevance by reranking initial results
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "rerank-english-v3.0"):
        """
        Args:
            api_key: Cohere API key (from env if None)
            model: Reranking model to use
        """
        if not COHERE_AVAILABLE:
            raise ImportError("Cohere package not installed. Install with: pip install cohere")
        
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            logger.warning("Cohere API key not found, reranking disabled")
            self.client = None
        else:
            self.client = cohere.Client(self.api_key)
            self.model = model
            logger.info(f"Reranker initialized with model: {model}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents based on query relevance
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_n: Number of top results to return (None = all)
            
        Returns:
            List of (document, relevance_score) tuples, sorted by relevance
        """
        if not self.client or not documents:
            return [(doc, 1.0) for doc in documents]
        
        try:
            # Cohere rerank API
            results = self.client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n or len(documents)
            )
            
            # Extract results
            reranked = [
                (documents[result.index], result.relevance_score)
                for result in results.results
            ]
            
            logger.debug(f"Reranked {len(documents)} documents, returning top {len(reranked)}")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback: return original order
            return [(doc, 1.0) for doc in documents]


class MultiHopRetriever:
    """
    Multi-hop retrieval system
    Performs iterative retrieval with query refinement
    """
    
    def __init__(self, knowledge_base, max_hops: int = 3):
        """
        Args:
            knowledge_base: Agno Knowledge instance
            max_hops: Maximum number of retrieval iterations
        """
        self.knowledge_base = knowledge_base
        self.max_hops = max_hops
        logger.info(f"Multi-hop retriever initialized with max_hops={max_hops}")
    
    def retrieve(
        self,
        initial_query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-hop retrieval
        
        Args:
            initial_query: Initial user query
            context: Additional context from conversation
            
        Returns:
            Dictionary with retrieved information and retrieval path
        """
        all_results = []
        current_query = initial_query
        retrieval_path = []
        
        for hop in range(self.max_hops):
            logger.debug(f"Retrieval hop {hop + 1}/{self.max_hops} with query: {current_query[:100]}...")
            
            # Perform retrieval
            try:
                # Use Agno's knowledge search
                if hasattr(self.knowledge_base, 'search'):
                    results = self.knowledge_base.search(current_query, limit=5)
                else:
                    # Fallback: direct vector search
                    results = []
                
                if results:
                    all_results.extend(results)
                    retrieval_path.append({
                        'hop': hop + 1,
                        'query': current_query,
                        'results_count': len(results)
                    })
                    
                    # Check if we have enough information
                    if len(all_results) >= 10:
                        break
                    
                    # Refine query for next hop
                    if hop < self.max_hops - 1:
                        current_query = self._refine_query(
                            initial_query,
                            current_query,
                            results,
                            context
                        )
                else:
                    # No more results, stop
                    break
                    
            except Exception as e:
                logger.error(f"Error in retrieval hop {hop + 1}: {e}")
                break
        
        return {
            'results': all_results[:10],  # Limit total results
            'retrieval_path': retrieval_path,
            'num_hops': len(retrieval_path)
        }
    
    def _refine_query(
        self,
        original_query: str,
        current_query: str,
        results: List[Any],
        context: Optional[str]
    ) -> str:
        """
        Refine query for next hop based on current results
        
        Args:
            original_query: Original user query
            current_query: Query used in current hop
            results: Results from current hop
            context: Additional context
            
        Returns:
            Refined query for next hop
        """
        # Extract key terms from results
        result_texts = []
        for result in results[:3]:  # Use top 3 results
            if hasattr(result, 'content'):
                result_texts.append(result.content[:200])
            elif isinstance(result, str):
                result_texts.append(result[:200])
        
        # Simple query expansion: add context from results
        if result_texts:
            # Combine original query with key information
            refined = f"{original_query} [Context from previous search: {' '.join(result_texts[:2])}]"
        else:
            refined = original_query
        
        return refined[:500]  # Limit length


class QueryExpander:
    """
    Query expansion for better retrieval
    Generates multiple query variations
    """
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for query generation
        """
        self.llm_client = llm_client
        logger.info("Query expander initialized")
    
    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        variations = [query]  # Always include original
        
        if self.llm_client:
            # Use LLM to generate variations
            try:
                prompt = f"""Generate {num_variations} different ways to search for this information:
Query: {query}

Generate variations that:
1. Use different terminology
2. Focus on different aspects
3. Use synonyms or related terms

Return only the variations, one per line:"""
                
                # This would use the LLM client
                # For now, use simple heuristics
                variations.extend(self._simple_expansion(query, num_variations))
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}, using simple expansion")
                variations.extend(self._simple_expansion(query, num_variations))
        else:
            variations.extend(self._simple_expansion(query, num_variations))
        
        return variations[:num_variations + 1]
    
    def _simple_expansion(self, query: str, num_variations: int) -> List[str]:
        """Simple query expansion using heuristics"""
        variations = []
        
        # Add "what is" variation
        if not query.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            variations.append(f"what is {query}")
        
        # Add "explain" variation
        if 'explain' not in query.lower():
            variations.append(f"explain {query}")
        
        # Add "information about" variation
        variations.append(f"information about {query}")
        
        return variations[:num_variations]


class EnhancedRAG:
    """
    Enhanced RAG system combining reranking, multi-hop retrieval, and query expansion
    """
    
    def __init__(
        self,
        knowledge_base,
        use_reranking: bool = True,
        use_multi_hop: bool = True,
        use_query_expansion: bool = False,
        cohere_api_key: Optional[str] = None
    ):
        """
        Args:
            knowledge_base: Agno Knowledge instance
            use_reranking: Enable Cohere reranking
            use_multi_hop: Enable multi-hop retrieval
            use_query_expansion: Enable query expansion
            cohere_api_key: Cohere API key for reranking
        """
        self.knowledge_base = knowledge_base
        
        # Initialize components
        self.reranker = None
        if use_reranking and COHERE_AVAILABLE:
            try:
                self.reranker = Reranker(api_key=cohere_api_key)
                logger.info("Reranking enabled")
            except Exception as e:
                logger.warning(f"Could not initialize reranker: {e}")
        
        self.multi_hop = None
        if use_multi_hop:
            self.multi_hop = MultiHopRetriever(knowledge_base, max_hops=3)
            logger.info("Multi-hop retrieval enabled")
        
        self.query_expander = None
        if use_query_expansion:
            self.query_expander = QueryExpander()
            logger.info("Query expansion enabled")
        
        logger.info("Enhanced RAG system initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced retrieval with all enabled features
        
        Args:
            query: User query
            top_k: Number of results to return
            context: Additional context
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        # Step 1: Query expansion (if enabled)
        queries = [query]
        if self.query_expander:
            queries = self.query_expander.expand(query, num_variations=2)
            logger.debug(f"Expanded query into {len(queries)} variations")
        
        # Step 2: Multi-hop retrieval (if enabled)
        all_results = []
        retrieval_metadata = {}
        
        if self.multi_hop:
            for q in queries[:1]:  # Use original query for multi-hop
                hop_results = self.multi_hop.retrieve(q, context)
                all_results.extend(hop_results['results'])
                retrieval_metadata = hop_results
        else:
            # Simple single-hop retrieval
            try:
                if hasattr(self.knowledge_base, 'search'):
                    for q in queries:
                        results = self.knowledge_base.search(q, limit=top_k)
                        all_results.extend(results)
                else:
                    logger.warning("Knowledge base does not support search")
            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
        
        # Step 3: Reranking (if enabled)
        if self.reranker and all_results:
            # Extract text from results
            doc_texts = []
            for result in all_results:
                if hasattr(result, 'content'):
                    doc_texts.append(result.content)
                elif isinstance(result, str):
                    doc_texts.append(result)
                else:
                    doc_texts.append(str(result))
            
            if doc_texts:
                reranked = self.reranker.rerank(query, doc_texts, top_n=top_k)
                
                # Reorder results based on reranking
                reranked_results = []
                reranked_indices = {text: idx for idx, (text, score) in enumerate(reranked)}
                
                for result in all_results:
                    result_text = str(result.content if hasattr(result, 'content') else result)
                    if result_text in reranked_indices:
                        reranked_results.append((result, reranked[reranked_indices[result_text]][1]))
                
                # Sort by rerank score
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                all_results = [r[0] for r in reranked_results[:top_k]]
                retrieval_metadata['reranked'] = True
                retrieval_metadata['rerank_scores'] = [r[1] for r in reranked_results[:top_k]]
        
        return {
            'results': all_results[:top_k],
            'metadata': retrieval_metadata,
            'num_results': len(all_results[:top_k])
        }

