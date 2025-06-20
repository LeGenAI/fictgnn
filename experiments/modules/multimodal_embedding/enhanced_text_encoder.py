"""
Enhanced Text Encoder for Financial Documents

Inspired by:
1. graphSEAT (CIKM'20): Fusing Global Domain Information and Local Semantic Information
2. MRG-for-Finance: Multi-Relational Graph for Finance
3. OpenAI text-embedding-3-small: Cost-effective high-performance embeddings

Key Features:
- Global Domain Context + Local Semantic Fusion
- Multi-relational financial knowledge integration  
- OpenAI text-embedding-3-small integration
- Hierarchical document processing
- Financial domain specialization
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# OpenAI imports
try:
    from openai import OpenAI
    import tiktoken
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logging.warning("OpenAI package not found. Install with: pip install openai")

# Sentence Transformers as fallback
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Retry mechanism
try:
    from tenacity import retry, wait_random_exponential, stop_after_attempt
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    logging.warning("Tenacity not found. Install with: pip install tenacity")

logger = logging.getLogger(__name__)


@dataclass
class FinancialDocument:
    """Financial document structure"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    company: Optional[str] = None
    industry: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = None  # report, news, filing, etc.
    

@dataclass
class DomainContext:
    """Global domain context for financial documents"""
    industry_keywords: Dict[str, List[str]] = field(default_factory=dict)
    financial_concepts: List[str] = field(default_factory=list)
    company_relations: Dict[str, List[str]] = field(default_factory=dict)
    market_segments: Dict[str, List[str]] = field(default_factory=dict)


class FinancialKnowledgeBase:
    """
    Financial domain knowledge base inspired by graphSEAT's global domain information
    """
    
    def __init__(self):
        self.domain_context = self._build_domain_context()
        self.relation_types = self._define_relation_types()
        
    def _build_domain_context(self) -> DomainContext:
        """Build comprehensive financial domain context"""
        return DomainContext(
            industry_keywords={
                "semiconductor": [
                    "semiconductor", "memory", "DRAM", "NAND", "SSD", "foundry", "wafer",
                    "AP", "GPU", "CPU", "AI chip", "HBM", "DDR", "NAND flash"
                ],
                "technology": [
                    "artificial intelligence", "AI", "cloud", "big data", "IoT", "5G", "6G",
                    "machine learning", "deep learning", "blockchain", "metaverse", "NFT"
                ],
                "automotive": [
                    "electric vehicle", "EV", "battery", "autonomous driving", "ADAS", "infotainment",
                    "automotive semiconductor", "LiDAR", "camera module"
                ],
                "display": [
                    "OLED", "LCD", "QLED", "microLED", "display panel",
                    "touch panel", "flexible", "foldable"
                ]
            },
            financial_concepts=[
                "revenue", "operating profit", "net profit", "EBITDA", "ROE", "ROA", "PER", "PBR",
                "debt ratio", "current ratio", "capital", "market cap", "dividend", "dividend yield",
                "growth rate", "margin", "cash flow", "return on investment"
            ],
            company_relations={
                "supply_chain": ["supplier", "partner", "partner", "customer"],
                "competition": ["competitor", "rival", "market share", "competitive advantage"],
                "investment": ["investment", "stake", "investment", "M&A", "M&A"]
            },
            market_segments={
                "B2B": ["enterprise customers", "industrial", "enterprise"],
                "B2C": ["individual consumers", "general consumers", "retail"],
                "government": ["government", "public", "national projects"]
            }
        )
    
    def _define_relation_types(self) -> Dict[str, Dict[str, Any]]:
        """Define multi-relational types inspired by MRG-for-Finance"""
        return {
            "industry_relation": {
                "description": "Same industry relationships between companies",
                "weight": 0.8,
                "keywords": ["same industry", "industry", "sector", "sectoral"]
            },
            "supply_chain_relation": {
                "description": "Supply chain relationships",
                "weight": 0.9,
                "keywords": ["supplier", "delivery", "procurement", "supply", "vendor"]
            },
            "investment_relation": {
                "description": "Investment and equity relationships", 
                "weight": 0.7,
                "keywords": ["investment", "stake", "investment", "stake", "investment"]
            },
            "technology_relation": {
                "description": "Technology cooperation relationships",
                "weight": 0.6,
                "keywords": ["technology cooperation", "patent", "licensing", "R&D", "joint"]
            },
            "market_relation": {
                "description": "Market and customer relationships",
                "weight": 0.5,
                "keywords": ["customer", "market", "sales", "customer", "market"]
            }
        }
    
    def extract_domain_features(self, document: FinancialDocument) -> Dict[str, float]:
        """Extract global domain features from document"""
        features = {}
        content_lower = document.content.lower()
        
        # Industry relevance scores
        for industry, keywords in self.domain_context.industry_keywords.items():
            score = sum(content_lower.count(keyword.lower()) for keyword in keywords)
            features[f"industry_{industry}"] = float(score)
        
        # Financial concept density
        fin_score = sum(content_lower.count(concept.lower()) 
                       for concept in self.domain_context.financial_concepts)
        features["financial_density"] = float(fin_score / max(len(content_lower.split()), 1))
        
        # Relation type scores
        for rel_type, rel_info in self.relation_types.items():
            score = sum(content_lower.count(keyword.lower()) 
                       for keyword in rel_info["keywords"])
            features[f"relation_{rel_type}"] = float(score)
        
        return features


class OpenAIEmbeddingEngine:
    """
    OpenAI text-embedding-3-small integration with best practices
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        max_tokens: int = 8191,
        batch_size: int = 100,
        max_retries: int = 6
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize OpenAI client
        if not HAS_OPENAI:
            raise ImportError("OpenAI package required. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # fallback
        
        logger.info(f"OpenAI embedding engine initialized with {model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str, max_chunk_tokens: int = 8000) -> List[str]:
        """Intelligently chunk text to fit token limits"""
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_chunk_tokens:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]\s+', text)
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens <= max_chunk_tokens:
                current_chunk += sentence + ". "
                current_tokens += sentence_tokens
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                current_tokens = sentence_tokens
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for single text with retry mechanism"""
        if HAS_TENACITY:
            # Use tenacity for retry
            @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
            def _get_embedding_with_retry():
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model_name
                )
                return response.data[0].embedding
            
            return _get_embedding_with_retry()
        else:
            # Simple retry without tenacity
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(
                        input=[text],
                        model=self.model_name
                    )
                    return response.data[0].embedding
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(min(2 ** attempt, 20))
            
            # Should not reach here
            raise RuntimeError("Max retries exceeded")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for batch of texts"""
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        
        if not valid_texts:
            return []
        
        # Process in batches
        embeddings = []
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            
            # Chunk texts that are too long
            processed_batch = []
            chunk_mapping = []  # Track which chunks belong to which original text
            
            for idx, text in enumerate(batch):
                chunks = self.chunk_text(text)
                processed_batch.extend(chunks)
                chunk_mapping.extend([idx] * len(chunks))
            
            # Get embeddings for processed batch
            try:
                response = self.client.embeddings.create(
                    input=processed_batch,
                    model=self.model_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                
                # Aggregate chunk embeddings back to original texts
                text_embeddings = []
                for idx in range(len(batch)):
                    # Find all chunks for this text
                    chunk_indices = [i for i, mapping_idx in enumerate(chunk_mapping) if mapping_idx == idx]
                    
                    if len(chunk_indices) == 1:
                        # Single chunk
                        text_embeddings.append(batch_embeddings[chunk_indices[0]])
                    else:
                        # Multiple chunks - average them
                        chunk_embeddings = [batch_embeddings[i] for i in chunk_indices]
                        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
                        text_embeddings.append(avg_embedding)
                
                embeddings.extend(text_embeddings)
                
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fallback to individual requests
                for text in batch:
                    try:
                        emb = self.get_embedding(text)
                        embeddings.append(emb)
                    except Exception as e2:
                        logger.error(f"Individual embedding failed: {e2}")
                        # Use zero embedding as fallback
                        embeddings.append([0.0] * 1536)
        
        return embeddings


class HierarchicalDocumentProcessor:
    """
    Hierarchical document processing inspired by graphSEAT's local semantic information
    """
    
    def __init__(self):
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^[\d\.\)]+\s+(.+)$',  # Numbered sections
            r'^[IVX]+\.\s+(.+)$',  # Roman numerals
        ]
    
    def extract_document_hierarchy(self, content: str) -> Dict[str, Any]:
        """Extract hierarchical structure from document"""
        lines = content.split('\n')
        hierarchy = {
            'title': '',
            'sections': [],
            'paragraphs': [],
            'tables': [],
            'key_sentences': []
        }
        
        current_section = None
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    hierarchy['paragraphs'].append(' '.join(current_paragraph))
                    current_paragraph = []
                continue
            
            # Check for section headers
            is_header = False
            for pattern in self.section_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    is_header = True
                    if current_paragraph:
                        hierarchy['paragraphs'].append(' '.join(current_paragraph))
                        current_paragraph = []
                    
                    # Save previous section
                    if current_section:
                        hierarchy['sections'].append(current_section)
                    
                    # Start new section
                    current_section = {
                        'title': match.group(1),
                        'content': '',
                        'level': len(line) - len(line.lstrip('#')) if line.startswith('#') else 1
                    }
                    break
            
            if not is_header:
                # Check for table
                if '|' in line and len(line.split('|')) > 2:
                    hierarchy['tables'].append(line)
                else:
                    # Regular content
                    current_paragraph.append(line)
                    if current_section:
                        current_section['content'] += line + ' '
        
        # Save last paragraph and section
        if current_paragraph:
            hierarchy['paragraphs'].append(' '.join(current_paragraph))
        if current_section:
            hierarchy['sections'].append(current_section)
        
        # Extract key sentences (first sentence of each paragraph)
        for paragraph in hierarchy['paragraphs']:
            sentences = re.split(r'[.!?]\s+', paragraph)
            if sentences and len(sentences[0]) > 20:
                hierarchy['key_sentences'].append(sentences[0])
        
        return hierarchy


class GraphSEATInspiredEncoder(nn.Module):
    """
    Text encoder inspired by graphSEAT architecture
    
    Combines:
    - Global domain information (financial knowledge)
    - Local semantic information (document-specific content)
    - Multi-relational awareness
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        use_openai: bool = True,
        fallback_model: str = "all-mpnet-base-v2",
        hidden_dim: int = 512,
        domain_weight: float = 0.3,
        semantic_weight: float = 0.7,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.use_openai = use_openai and HAS_OPENAI
        self.device = device
        self.domain_weight = domain_weight
        self.semantic_weight = semantic_weight
        
        # Initialize components
        self.knowledge_base = FinancialKnowledgeBase()
        self.doc_processor = HierarchicalDocumentProcessor()
        
        # Initialize embedding engines
        if self.use_openai:
            try:
                self.openai_engine = OpenAIEmbeddingEngine(api_key=openai_api_key)
                self.embedding_dim = 1536  # text-embedding-3-small dimension
                logger.info("Using OpenAI text-embedding-3-small")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}, falling back to SentenceTransformers")
                self.use_openai = False
        
        if not self.use_openai:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("Either OpenAI or SentenceTransformers required")
            self.fallback_model = SentenceTransformer(fallback_model, device=device)
            self.embedding_dim = self.fallback_model.get_sentence_embedding_dimension()
            logger.info(f"Using fallback model: {fallback_model}")
        
        # Neural network components
        self.domain_feature_dim = 20  # Number of domain features
        
        # Domain information processor
        self.domain_processor = nn.Sequential(
            nn.Linear(self.domain_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        ).to(device)
        
        # Semantic information processor
        self.semantic_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        
        # Cross-attention for global-local fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        ).to(device)
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        ).to(device)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.embedding_dim).to(device)
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get raw text embeddings"""
        if self.use_openai:
            embeddings = self.openai_engine.get_embeddings_batch(texts)
            return torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        else:
            embeddings = self.fallback_model.encode(
                texts, 
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )
            return embeddings.to(self.device)
    
    def extract_hierarchical_content(self, documents: List[FinancialDocument]) -> Dict[str, List[str]]:
        """Extract hierarchical content from documents"""
        content_types = {
            'full_content': [],
            'sections': [],
            'key_sentences': [],
            'abstracts': []
        }
        
        for doc in documents:
            hierarchy = self.doc_processor.extract_document_hierarchy(doc.content)
            
            # Full content
            content_types['full_content'].append(doc.content)
            
            # Section summaries
            section_summary = ' '.join([
                f"{section['title']}: {section['content'][:200]}..." 
                for section in hierarchy['sections'][:3]
            ])
            if not section_summary.strip():
                section_summary = doc.content[:300]  # Fallback to content
            content_types['sections'].append(section_summary)
            
            # Key sentences
            key_summary = ' '.join(hierarchy['key_sentences'][:5])
            if not key_summary.strip():
                key_summary = doc.content[:200]  # Fallback to content
            content_types['key_sentences'].append(key_summary)
            
            # Abstract (first paragraph)
            abstract = hierarchy['paragraphs'][0] if hierarchy['paragraphs'] else doc.content[:300]
            if not abstract.strip():
                abstract = doc.content[:300]  # Fallback to content
            content_types['abstracts'].append(abstract)
        
        return content_types
    
    def forward(self, documents: List[FinancialDocument]) -> torch.Tensor:
        """
        Forward pass: Global domain + Local semantic fusion
        
        Args:
            documents: List of FinancialDocument objects
            
        Returns:
            Fused embeddings combining global and local information
        """
        batch_size = len(documents)
        
        # 1. Extract global domain features
        domain_features = []
        for doc in documents:
            features = self.knowledge_base.extract_domain_features(doc)
            # Convert to fixed-size vector
            feature_vector = [
                features.get(f"industry_semiconductor", 0),
                features.get(f"industry_technology", 0),
                features.get(f"industry_automotive", 0),
                features.get(f"industry_display", 0),
                features.get("financial_density", 0),
                features.get("relation_industry_relation", 0),
                features.get("relation_supply_chain_relation", 0),
                features.get("relation_investment_relation", 0),
                features.get("relation_technology_relation", 0),
                features.get("relation_market_relation", 0)
            ]
            # Pad to domain_feature_dim
            while len(feature_vector) < self.domain_feature_dim:
                feature_vector.append(0.0)
            domain_features.append(feature_vector[:self.domain_feature_dim])
        
        domain_tensor = torch.tensor(domain_features, dtype=torch.float32, device=self.device)
        
        # 2. Extract local semantic information (hierarchical)
        content_hierarchy = self.extract_hierarchical_content(documents)
        
        # Get embeddings for different content levels
        full_embeddings = self.get_text_embeddings(content_hierarchy['full_content'])
        section_embeddings = self.get_text_embeddings(content_hierarchy['sections'])
        key_embeddings = self.get_text_embeddings(content_hierarchy['key_sentences'])
        abstract_embeddings = self.get_text_embeddings(content_hierarchy['abstracts'])
        
        # Handle empty embeddings by using full_embeddings as fallback
        if section_embeddings.numel() == 0:
            section_embeddings = full_embeddings
        if key_embeddings.numel() == 0:
            key_embeddings = full_embeddings
        if abstract_embeddings.numel() == 0:
            abstract_embeddings = full_embeddings
        
        # Weighted combination of hierarchical embeddings
        semantic_embeddings = (
            0.4 * full_embeddings +
            0.3 * section_embeddings +
            0.2 * key_embeddings +
            0.1 * abstract_embeddings
        )
        
        # 3. Process domain and semantic information
        domain_processed = self.domain_processor(domain_tensor)  # [batch, hidden//4]
        semantic_processed = self.semantic_processor(semantic_embeddings)  # [batch, hidden]
        
        # 4. Cross-attention fusion (global-local interaction)
        # Add sequence dimension for attention
        semantic_seq = semantic_processed.unsqueeze(1)  # [batch, 1, hidden]
        domain_seq = domain_processed.unsqueeze(1)  # [batch, 1, hidden//4]
        
        # Pad domain to match semantic dimension for attention
        domain_padded = F.pad(domain_seq, (0, semantic_processed.size(-1) - domain_processed.size(-1)))
        
        # Apply cross-attention
        attended_semantic, _ = self.cross_attention(
            semantic_seq, domain_padded, domain_padded
        )
        attended_semantic = attended_semantic.squeeze(1)  # [batch, hidden]
        
        # 5. Final fusion
        fused_features = torch.cat([attended_semantic, domain_processed], dim=-1)
        fused_output = self.fusion_layer(fused_features)
        
        # 6. Project to output dimension
        final_embeddings = self.output_projection(fused_output)
        
        return final_embeddings
    
    def encode_documents(self, documents: List[FinancialDocument]) -> torch.Tensor:
        """Convenience method for encoding documents"""
        with torch.no_grad():
            return self.forward(documents)
    
    def encode_texts(self, texts: List[str], **metadata) -> torch.Tensor:
        """Encode raw texts with optional metadata"""
        documents = []
        for text in texts:
            doc = FinancialDocument(content=text, metadata=metadata)
            documents.append(doc)
        
        return self.encode_documents(documents)
    
    def get_embedding_dimension(self) -> int:
        """Get output embedding dimension"""
        return self.embedding_dim
    
    def load_model(self) -> bool:
        """
        Load the text encoder model
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            logger.info("Loading GraphSEAT text encoder model...")
            
            # OpenAI models don't require separate loading
            if self.use_openai:
                logger.info("OpenAI model ready")
                return True
            
            # Test SentenceTransformers model
            if hasattr(self, 'fallback_model') and self.fallback_model is not None:
                # Run simple test
                test_result = self.fallback_model.encode(
                    ["Test text"], 
                    convert_to_tensor=True, 
                    device=self.device,
                    show_progress_bar=False
                )
                if test_result is not None and test_result.numel() > 0:
                    logger.info("SentenceTransformers model loading successful")
                    return True
                else:
                    logger.error("SentenceTransformers model test failed")
                    return False
            else:
                logger.error("No available text model")
                return False
                
        except Exception as e:
            logger.error(f"Text encoder model loading failed: {e}")
            return False