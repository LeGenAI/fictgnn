#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causality Analyzer for Semiconductor Industry Analysis

A comprehensive causal relationship analysis module for semiconductor industry
research using advanced graph neural networks and industry knowledge.

Author: FicTGNN Research Team
Date: 2024
Version: 2.0 (Research Framework)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Conditional absolute imports for flexibility
try:
    from config.industry_knowledge import SEMICONDUCTOR_INDUSTRY_KNOWLEDGE
    from utils.logging_utils import LoggerSetup, StructuredLogger
except ImportError:
    try:
        from semiconductor_analysis.config.industry_knowledge import SEMICONDUCTOR_INDUSTRY_KNOWLEDGE
        from semiconductor_analysis.utils.logging_utils import LoggerSetup, StructuredLogger
    except ImportError:
        # Fallback dummy data for testing environments
        SEMICONDUCTOR_INDUSTRY_KNOWLEDGE = {
            'supply_chain_relationships': {},
            'verified_relationships': {'revenue_dependency': {}},
            'corporate_groups': {},
            'industry_patterns': {'memory_semiconductor': {}, 'display': {}},
            'supply_chain_trends': {},
            'metadata': {}
        }

        class StructuredLogger:
            def __init__(self, logger):
                self.logger = logger
            def log_operation_start(self, *args, **kwargs):
                self.logger.info(f"Operation started: {args[0] if args else 'Unknown'}")
            def log_operation_end(self, *args, **kwargs):
                self.logger.info(f"Operation completed: {args[0] if args else 'Unknown'}")
            def log_progress(self, current, total, operation=""):
                self.logger.info(f"Progress: {current}/{total} ({current/total*100:.1f}%) - {operation}")


class CausalityAnalyzer:
    """
    Advanced Causality Analysis Framework for Semiconductor Industry
    
    This class implements comprehensive causal relationship analysis using
    industry-aware embeddings and graph neural network architectures.
    
    Args:
        config (Dict[str, Any]): Configuration parameters for the analyzer
        
    Attributes:
        config: Analysis configuration parameters
        logger: Structured logging interface
        default_threshold: Default causality detection threshold
        max_companies: Maximum companies per analysis
        max_hops: Maximum propagation hops for influence analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize structured logger
        logger = LoggerSetup.setup_logger(
            name='causality_analyzer',
            level='INFO',
            filename='logs/causality_analysis.log'
        )
        self.structured_logger = StructuredLogger(logger)
        
        # Analysis parameters
        self.default_threshold = config.get('analysis', {}).get('causality_threshold', 0.8)
        self.max_companies = config.get('analysis', {}).get('max_companies_per_analysis', 10)
        self.max_hops = config.get('analysis', {}).get('max_propagation_hops', 3)
        
        self.structured_logger.log_operation_start("Causality analyzer initialization")
    
    def initialize_with_data(self, model: Any, graph_data: Any, node_mapping: Dict[Any, int], 
                           reverse_mapping: Dict[int, Any], enhanced_embeddings: np.ndarray):
        """
        Initialize analyzer with data components
        
        Args:
            model: Trained GNN model
            graph_data: Graph structure data
            node_mapping: Node to index mapping
            reverse_mapping: Index to node mapping
            enhanced_embeddings: Enhanced node embeddings
        """
        self.model = model
        self.graph_data = graph_data
        self.node_mapping = node_mapping
        self.reverse_mapping = reverse_mapping
        self.enhanced_embeddings = enhanced_embeddings
        
        self.structured_logger.log_operation_end("Analysis data initialization",
                                                nodes=len(self.reverse_mapping),
                                                embedding_dim=self.enhanced_embeddings.shape[1],
                                                edges=len(self.graph_data.edge_attr) if hasattr(self.graph_data, 'edge_attr') else 0)
        
        self.logger.info(f"Analysis data initialized - Nodes: {len(self.reverse_mapping)}, "
                        f"Embedding dimension: {self.enhanced_embeddings.shape[1]}, "
                        f"Edges: {len(self.graph_data.edge_attr) if hasattr(self.graph_data, 'edge_attr') else 0}")
    
    def calculate_industry_aware_similarity(self, source_company: str, target_company: str, 
                                          source_emb: np.ndarray, target_emb: np.ndarray,
                                          source_quarter: str, target_quarter: str) -> float:
        """
        Calculate industry-aware similarity between companies using multi-modal embeddings
        
        This method computes similarity by considering textual embeddings, financial features,
        industry-specific knowledge, and temporal relationships.
        
        Args:
            source_company: Source company name
            target_company: Target company name
            source_emb: Source company embedding vector
            target_emb: Target company embedding vector
            source_quarter: Source quarter identifier
            target_quarter: Target quarter identifier
            
        Returns:
            float: Industry-aware similarity score (0-1)
        """
        
        # Dynamic dimension extraction
        total_dims = len(source_emb)
        industry_dims = 8  # Industry-specific features (8 dimensions)
        
        # Estimate dimension allocation
        if total_dims > 30:
            text_dims = total_dims - industry_dims - 10
            financial_dims = 10
        else:
            text_dims = max(12, total_dims - industry_dims - 8)
            financial_dims = total_dims - text_dims - industry_dims
        
        # Dimension-wise feature separation
        source_text = source_emb[:text_dims]
        target_text = target_emb[:text_dims]
        source_financial = source_emb[text_dims:text_dims+financial_dims]
        target_financial = target_emb[text_dims:text_dims+financial_dims]
        source_industry = source_emb[text_dims+financial_dims:]
        target_industry = target_emb[text_dims+financial_dims:]
        
        # Similarity computation across modalities
        text_similarity = cosine_similarity([source_text], [target_text])[0][0]
        financial_similarity = cosine_similarity([source_financial], [target_financial])[0][0]
        industry_similarity = cosine_similarity([source_industry], [target_industry])[0][0]
        
        # Industry knowledge-based relationship scoring
        industry_score = 0.0
        supply_chain_data = SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['supply_chain_relationships']
        
        # Supply chain relationship analysis
        if target_company in supply_chain_data.get(source_company, {}).get('suppliers', []):
            industry_score += 0.8  # Strong supply chain relationship
        elif source_company in supply_chain_data.get(target_company, {}).get('suppliers', []):
            industry_score += 0.6  # Reverse supply chain relationship
        
        # Temporal discounting
        try:
            source_year, source_q = source_quarter.split('Q')
            target_year, target_q = target_quarter.split('Q')
            time_diff = (int(target_year) - int(source_year)) * 4 + (int(target_q) - int(source_q))
            time_discount = max(0.1, 1.0 - 0.1 * time_diff)
        except:
            time_discount = 1.0
        
        # Weighted final similarity calculation
        text_weight = 0.5      # 50% - Textual embeddings (primary)
        financial_weight = 0.3  # 30% - Financial similarity
        industry_sim_weight = 0.1  # 10% - Industry embedding similarity
        industry_rel_weight = 0.1  # 10% - Industry relationship knowledge
        
        final_similarity = (
            text_weight * text_similarity + 
            financial_weight * financial_similarity + 
            industry_sim_weight * industry_similarity +
            industry_rel_weight * min(industry_score, 1.0)
        ) * time_discount
        
        return final_similarity
    
    def extract_causal_statements(self, top_k: int = 30) -> List[str]:
        """
        Extract key causal relationship statements from graph data
        
        Args:
            top_k: Number of top causal statements to extract
            
        Returns:
            List[str]: List of formatted causal statements
        """
        statements = []
        
        if not hasattr(self.graph_data, 'edge_attr'):
            self.logger.warning("Graph data missing edge attributes")
            return statements
        
        # Filter causality edges
        causal_edges = []
        for i, attr in enumerate(self.graph_data.edge_attr):
            if attr.get('type') == 'industry_causal':
                causal_edges.append((i, attr))
        
        # Sort by weight
        causal_edges.sort(key=lambda x: x[1].get('weight', 0), reverse=True)
        
        # Generate top-k statements
        for i, (edge_idx, attr) in enumerate(causal_edges[:top_k]):
            source_company = attr.get('source_company', 'Unknown')
            target_company = attr.get('target_company', 'Unknown')
            source_quarter = attr.get('source_quarter', 'Unknown')
            target_quarter = attr.get('target_quarter', 'Unknown')
            weight = attr.get('weight', 0.0)
            similarity = attr.get('similarity', 0.0)
            lag = attr.get('lag', 0)
            
            statement = (f"{source_quarter} {source_company} → "
                        f"{target_quarter} {target_company} "
                        f"(influence: {weight:.3f}, similarity: {similarity:.3f}, lag: {lag}Q)")
            
            statements.append(statement)
        
        return statements
    
    def analyze_all_companies_influence(self, influence_threshold: Optional[float] = None, 
                                     max_companies: Optional[int] = None) -> Dict[str, Any]:
        """
        Comprehensive influence analysis across all companies with directional edge generation
        
        This method performs systematic influence analysis between all company pairs,
        identifying significant causal relationships based on industry-aware similarity.
        
        Args:
            influence_threshold: Minimum influence threshold for relationship detection
            max_companies: Maximum number of companies per influence analysis
            
        Returns:
            Dict[str, Any]: Comprehensive influence analysis results
        """
        
        if influence_threshold is None:
            influence_threshold = self.default_threshold
        if max_companies is None:
            max_companies = self.max_companies
            
        self.structured_logger.log_operation_start("Company-wide influence analysis", threshold=influence_threshold)
        
        all_influences = {}
        directional_edges = []
        
        total_pairs = len(self.reverse_mapping) * (len(self.reverse_mapping) - 1)
        processed = 0
        
        # Analyze influence patterns across all company-quarter combinations
        for source_idx, (source_company, source_quarter) in self.reverse_mapping.items():
            source_emb = self.enhanced_embeddings[source_idx]
            influences = []
            
            # Calculate similarity with all other nodes
            for target_idx, (target_company, target_quarter) in self.reverse_mapping.items():
                if source_idx == target_idx or source_company == target_company:
                    continue
                    
                processed += 1
                if processed % 10000 == 0:
                    self.structured_logger.log_progress(processed, total_pairs, "influence analysis")
                    
                target_emb = self.enhanced_embeddings[target_idx]
                similarity = self.calculate_industry_aware_similarity(
                    source_company, target_company, source_emb, target_emb, 
                    source_quarter, target_quarter
                )
                
                # Add significant relationships as directional edges
                if similarity > influence_threshold:
                    influences.append({
                        'target_company': target_company,
                        'target_quarter': target_quarter,
                        'similarity': similarity,
                        'target_idx': target_idx
                    })
                    
                    directional_edges.append({
                        'source': f"{source_quarter} {source_company}",
                        'target': f"{target_quarter} {target_company}",
                        'similarity': similarity,
                        'source_idx': source_idx,
                        'target_idx': target_idx
                    })
            
            # Store top influences only
            influences.sort(key=lambda x: x['similarity'], reverse=True)
            if influences:
                all_influences[f"{source_quarter} {source_company}"] = influences[:max_companies]
        
        # Identify top influencer companies
        top_influencers = []
        for source, influences in all_influences.items():
            if influences:
                avg_influence = np.mean([inf['similarity'] for inf in influences])
                top_influencers.append({
                    'company': source,
                    'influenced_count': len(influences),
                    'avg_influence': avg_influence,
                    'max_influence': max([inf['similarity'] for inf in influences]),
                    'influences': influences
                })
        
        # Sort by influence strength
        top_influencers.sort(key=lambda x: x['avg_influence'], reverse=True)
        
        self.structured_logger.log_operation_end("Influence analysis", 
                                                directional_edges=len(directional_edges),
                                                top_influencers=len(top_influencers))
        
        return {
            'threshold': influence_threshold,
            'total_directional_edges': len(directional_edges),
            'top_influencers': top_influencers[:20],  # Top 20 companies
            'directional_edges': directional_edges,
            'analysis_summary': {
                'companies_analyzed': len(all_influences),
                'companies_with_influence': len([k for k, v in all_influences.items() if v]),
                'avg_influences_per_company': np.mean([len(v) for v in all_influences.values()]) if all_influences else 0
            }
        }
    
    def analyze_influence_propagation(self, source_company: str, source_quarter: str, 
                                    max_hops: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze influence propagation patterns using breadth-first search
        
        This method traces how influence propagates through the network from a source
        company across multiple degrees of separation.
        
        Args:
            source_company: Source company for propagation analysis
            source_quarter: Source quarter for analysis
            max_hops: Maximum number of propagation hops to analyze
            
        Returns:
            Dict[str, Any]: Influence propagation analysis results
        """
        
        if max_hops is None:
            max_hops = self.max_hops
            
        self.logger.info(f"Analyzing influence propagation: {source_quarter} {source_company}")
        
        influenced_nodes = []
        
        # BFS for influence propagation tracking
        queue = [(source_company, source_quarter, 0)]
        visited = set()
        
        while queue:
            company, quarter, hop = queue.pop(0)
            
            if hop > max_hops:
                continue
                
            node_key = (company, quarter)
            if node_key in visited or node_key not in self.node_mapping:
                continue
                
            visited.add(node_key)
            
            # Find outgoing edges from current node
            source_idx = self.node_mapping[node_key]
            
            if hasattr(self.graph_data, 'edge_attr'):
                for i, attr in enumerate(self.graph_data.edge_attr):
                    if (attr.get('type') == 'industry_causal' and 
                        attr.get('source_company') == company and 
                        attr.get('source_quarter') == quarter):
                        
                        target_company = attr.get('target_company')
                        target_quarter = attr.get('target_quarter')
                        
                        influenced_nodes.append({
                            'company': target_company,
                            'quarter': target_quarter,
                            'hop': hop + 1,
                            'influence_score': attr.get('weight', 0.0),
                            'similarity': attr.get('similarity', 0.0)
                        })
                        
                        queue.append((target_company, target_quarter, hop + 1))
        
        # Group by hop distance
        influence_chain = defaultdict(list)
        for node in influenced_nodes:
            influence_chain[node['hop']].append({
                'company': node['company'],
                'quarter': node['quarter'],
                'influence_score': node['influence_score'],
                'similarity': node['similarity']
            })
        
        # Sort results by influence score
        for hop in influence_chain:
            influence_chain[hop].sort(key=lambda x: x['influence_score'], reverse=True)
        
        self.logger.info(f"Propagation analysis completed: {len(influenced_nodes)} nodes influenced")
        
        return {
            'source': f"{source_quarter} {source_company}",
            'total_influenced': len(influenced_nodes),
            'influence_chain': dict(influence_chain),
            'max_hops_analyzed': max_hops
        }
    
    def analyze_causality(self, graph: Any = None, model: Any = None) -> Dict[str, Any]:
        """
        Comprehensive causality analysis pipeline
        
        Args:
            graph: Optional graph data override
            model: Optional model override
            
        Returns:
            Dict[str, Any]: Complete causality analysis results
        """
        
        if graph is not None:
            self.graph_data = graph
        if model is not None:
            self.model = model
            
        self.logger.info("Executing comprehensive causality analysis")
        
        results = {}
        
        # Extract key causal statements
        causal_statements = self.extract_causal_statements(top_k=20)
        results['causal_statements'] = causal_statements
        
        # Company-wide influence analysis
        influence_analysis = self.analyze_all_companies_influence()
        results['influence_analysis'] = influence_analysis
        
        # Propagation analysis for top companies
        propagation_analyses = {}
        top_companies = influence_analysis['top_influencers'][:5]  # Top 5 companies
        
        for company_info in top_companies:
            company_name = company_info['company']
            # Parse company name and quarter
            if ' ' in company_name:
                parts = company_name.split(' ')
                quarter = parts[0]
                company = ' '.join(parts[1:])
                
                propagation = self.analyze_influence_propagation(company, quarter)
                propagation_analyses[company_name] = propagation
        
        results['propagation_analyses'] = propagation_analyses
        
        # Generate causality matrix
        causality_matrix = self._create_causality_matrix()
        results['causality_matrix'] = causality_matrix
        
        self.logger.info("Comprehensive causality analysis completed")
        
        return results
    
    def _create_causality_matrix(self) -> Dict[str, Any]:
        """
        Generate causality relationship matrix for companies
        
        Returns:
            Dict[str, Any]: Causality matrix data structure
        """
        
        # Company-to-index mapping
        companies = list(set([company for company, _ in self.reverse_mapping.values()]))
        company_to_idx = {company: idx for idx, company in enumerate(companies)}
        
        # Initialize matrix
        matrix_size = len(companies)
        causality_matrix = np.zeros((matrix_size, matrix_size))
        
        # Map edge information to matrix
        if hasattr(self.graph_data, 'edge_attr'):
            for attr in self.graph_data.edge_attr:
                if attr.get('type') == 'industry_causal':
                    source_company = attr.get('source_company')
                    target_company = attr.get('target_company')
                    weight = attr.get('weight', 0.0)
                    
                    if source_company in company_to_idx and target_company in company_to_idx:
                        source_idx = company_to_idx[source_company]
                        target_idx = company_to_idx[target_company]
                        causality_matrix[source_idx, target_idx] = max(
                            causality_matrix[source_idx, target_idx], weight
                        )
        
        return {
            'matrix': causality_matrix.tolist(),
            'companies': companies,
            'company_mapping': company_to_idx,
            'matrix_shape': causality_matrix.shape
        }
    
    def extract_key_insights(self, analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract key insights from causality analysis results
        
        Args:
            analysis_results: Optional pre-computed analysis results
            
        Returns:
            Dict[str, Any]: Structured key insights
        """
        
        if analysis_results is None:
            analysis_results = self.analyze_causality()
        
        self.logger.info("Extracting key insights from analysis results")
        
        insights = {
            'top_influencers': [],
            'emerging_patterns': [],
            'risk_indicators': [],
            'supply_chain_insights': [],
            'temporal_patterns': []
        }
        
        # Extract top influencer companies
        if 'influence_analysis' in analysis_results:
            top_influencers = analysis_results['influence_analysis']['top_influencers'][:5]
            for influencer in top_influencers:
                insights['top_influencers'].append({
                    'company': influencer['company'],
                    'influence_score': influencer['avg_influence'],
                    'affected_companies': influencer['influenced_count']
                })
        
        # Detect emerging patterns
        supply_chain_data = SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['supply_chain_relationships']
        for company in ['Samsung Electronics', 'SK Hynix', 'LG Innotek']:
            if company in supply_chain_data:
                suppliers = supply_chain_data[company].get('suppliers', [])
                insights['emerging_patterns'].append({
                    'pattern': f"{company} supply chain",
                    'suppliers': suppliers[:3],  # Top 3 suppliers
                    'supplier_count': len(suppliers)
                })
        
        # Risk indicator analysis
        verified_relations = SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['verified_relationships']
        for relation_type, relations in verified_relations['revenue_dependency'].items():
            company, client = relation_type
            dependency = relations.get('dependency', '')
            if '85%' in dependency or '90%' in dependency:
                insights['risk_indicators'].append({
                    'type': 'high_dependency',
                    'company': company,
                    'client': client,
                    'dependency': dependency,
                    'risk_level': 'high'
                })
        
        # Supply chain insights
        corporate_groups = SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['corporate_groups']
        for group_name, group_companies in corporate_groups.items():
            insights['supply_chain_insights'].append({
                'group': group_name,
                'companies': list(group_companies.keys()),
                'synergy_potential': len(group_companies) * 0.2  # Simple synergy score
            })
        
        # Temporal pattern analysis
        trends = SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['supply_chain_trends']
        for trend_key, trend_desc in trends.items():
            insights['temporal_patterns'].append({
                'pattern': trend_key,
                'description': trend_desc,
                'impact': 'medium'  # Default impact level
            })
        
        self.logger.info("Key insights extraction completed")
        
        return insights
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Returns:
            Dict[str, Any]: Complete analysis report with insights and recommendations
        """
        
        self.logger.info("Generating comprehensive analysis report")
        
        # Execute full analysis pipeline
        analysis_results = self.analyze_causality()
        
        # Extract key insights
        key_insights = self.extract_key_insights(analysis_results)
        
        # Compose final report
        report = {
            'analysis_summary': {
                'total_causal_relationships': len(analysis_results.get('causal_statements', [])),
                'total_companies_analyzed': len(self.reverse_mapping),
                'total_influence_edges': analysis_results.get('influence_analysis', {}).get('total_directional_edges', 0),
                'analysis_threshold': self.default_threshold
            },
            'detailed_results': analysis_results,
            'key_insights': key_insights,
            'recommendations': self._generate_recommendations(key_insights),
            'metadata': {
                'analysis_date': '2024-06-20',
                'version': '2.0',
                'framework': 'FicTGNN',
                'data_sources': SEMICONDUCTOR_INDUSTRY_KNOWLEDGE['metadata']
            }
        }
        
        self.logger.info("Comprehensive analysis report generated successfully")
        
        return report
    
    def _generate_recommendations(self, insights: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate actionable recommendations based on insights
        
        Args:
            insights: Extracted key insights
            
        Returns:
            List[Dict[str, str]]: Structured recommendations
        """
        
        recommendations = []
        
        # High-risk dependency recommendations
        for risk in insights.get('risk_indicators', []):
            if risk.get('risk_level') == 'high':
                recommendations.append({
                    'category': 'risk_management',
                    'priority': 'high',
                    'recommendation': f"Diversify {risk['company']} dependency on {risk['client']} ({risk['dependency']})"
                })
        
        # Supply chain optimization recommendations
        for group_insight in insights.get('supply_chain_insights', []):
            if group_insight.get('synergy_potential', 0) > 0.5:
                recommendations.append({
                    'category': 'supply_chain_optimization',
                    'priority': 'medium',
                    'recommendation': f"Explore synergy opportunities within {group_insight['group']} group"
                })
        
        # Technology trend adaptation
        for pattern in insights.get('temporal_patterns', []):
            if 'localization' in pattern.get('pattern', ''):
                recommendations.append({
                    'category': 'strategic_planning',
                    'priority': 'medium',
                    'recommendation': 'Strengthen domestic supply chain investments following localization trends'
                })
        
        return recommendations