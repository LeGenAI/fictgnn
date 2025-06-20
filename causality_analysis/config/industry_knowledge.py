#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semiconductor Industry Knowledge Database

Comprehensive domain knowledge database for the semiconductor industry based on
verified public disclosures and industry reports for academic research in
financial network analysis.

Author: Research Team
Date: 2024
Version: 2.0 (Modularized)
"""

# Semiconductor industry domain knowledge definition (June 2024 objective data)
SEMICONDUCTOR_INDUSTRY_KNOWLEDGE = {
    # Verified supply chain relationships (based on public disclosures and news)
    'supply_chain_relationships': {
        'Samsung Electronics': {
            'suppliers': ['Samsung Electro-Mechanics', 'LG Innotek', 'Wonik IPS', 'Dongjin Semichem', 'Solvrain', 
                         'Nepes', 'Nepes Ark', 'Simmtech', 'Semes', 'Hansol Technics'],
            'supplier_details': {
                'Samsung Electro-Mechanics': {'products': ['MLCCs', 'Camera modules', 'FC-BGA substrates'], 'relationship': 'Samsung Group affiliate'},
                'LG Innotek': {'products': ['CoF solutions', 'Camera modules'], 'contract': 'Galaxy A52 CoF exclusive supply 2021'},
                'Wonik IPS': {'products': ['Semiconductor equipment', 'Deposition equipment', 'ALD equipment'], 'ownership': 'Samsung Electronics + Samsung Display 7% stake'},
                'Dongjin Semichem': {'products': ['Photoresist', 'CMP slurry', 'Wet chemicals'], 'achievement': 'World 4th photoresist development 1989'},
                'Solvrain': {'products': ['High purity chemicals', 'Etching solutions', 'Cleaning chemicals'], 'cooperation': 'Joint localization of Japan export restriction items'},
                'Nepes': {'products': ['Semiconductor packaging', 'System semiconductor backend'], 'dependency': 'Key customer'},
                'Nepes Ark': {'products': ['PMIC testing', 'OLED DDI testing'], 'dependency': 'PMIC volume 85%+ revenue'}
            }
        },
        'SK Hynix': {
            'suppliers': ['Wonik IPS', 'Dongjin Semichem', 'Solvrain', 'Simmtech', 'Silicon Works', 
                         'SFA Engineering', 'Hana Micron', 'Doosan Tesna', 'Hanmi Semiconductor', 'APTC'],
            'supplier_details': {
                'Wonik IPS': {'products': ['ALD equipment', 'Semiconductor equipment'], 'trend': 'Revenue increase from 2024 customer diversification'},
                'Hana Micron': {'products': ['Memory semiconductor packaging', 'DRAM packaging', 'NAND packaging'], 'dependency': '60-70% revenue'},
                'SFA Engineering': {'products': ['Memory packaging', 'Flash memory', 'DRAM modules'], 'relationship': 'Major tier-1 supplier'},
                'Doosan Tesna': {'products': ['System semiconductor testing', 'Wafer probe testing'], 'relationship': 'Major customer'},
                'APTC': {'products': ['Metal etching equipment'], 'market_share': '90% share in SK Hynix'},
                'Hanmi Semiconductor': {'products': ['TC bonder for HBM'], 'relationship': 'HBM equipment supplier'}
            }
        },
        'LG Innotek': {
            'suppliers': ['BCNC', 'Duksan Neolux', 'Koh Young'],
            'customers': ['Samsung Electronics', 'Apple'],
            'customer_details': {
                'Apple': {'products': ['iPhone camera modules'], 'revenue_share': 'Majority of revenue'},
                'Samsung Electronics': {'products': ['CoF solutions'], 'contract': 'Galaxy A series'}
            }
        }
    },
    
    # Corporate group structures (verified affiliate relationships)
    'corporate_groups': {
        'Wonik Holdings': {
            'Wonik IPS': {'business': 'Semiconductor equipment', 'customers': ['Samsung Electronics', 'SK Hynix']},
            'Wonik QnC': {'business': 'Quartz parts and materials', 'customers': ['TSMC', 'GlobalFoundries'], 'acquisition': 'Momentive quartz business acquisition 2020'},
            'Wonik Materials': {'business': 'Semiconductor materials supply'}
        },
        'Nepes Group': {
            'Nepes': {'business': 'Semiconductor packaging', 'customers': ['Samsung Electronics', 'LG Display']},
            'Nepes Ark': {'business': 'Semiconductor testing', 'establishment': 'April 2019 spin-off from Nepes', 'revenue_source': 'Samsung Electronics PMIC testing 85%+'},
            'Nepes Lawe': {'business': 'FO-PLP', 'establishment': 'February 2020 establishment', 'status': 'Continuous losses'}
        },
        'Duksan Group': {
            'Duksan Neolux': {'business': 'OLED organic materials', 'customers': ['Samsung Display', 'LG Display'], 'position': 'Global leading OLED materials company'},
            'Duksan Tecopia': {'business': 'Semiconductor precursors OLED intermediates', 'relationship': 'OLED materials supply from Duksan Neolux'},
            'Duksan Hi-Metal': {'business': 'FC-BGA solder materials tin production'}
        }
    },
    
    # Industry supply chain patterns
    'industry_patterns': {
        'Memory Semiconductors': {
            'major_customers': ['Samsung Electronics', 'SK Hynix'],
            'equipment_suppliers': ['Wonik IPS', 'Jusung Engineering', 'Hanmi Semiconductor'],
            'materials_suppliers': ['Dongjin Semichem', 'Solvrain', 'Huchems'],
            'packaging_testing': ['Nepes', 'Hana Micron', 'SFA Engineering', 'Doosan Tesna']
        },
        'Display': {
            'major_customers': ['Samsung Display', 'LG Display'],
            'materials_suppliers': ['Duksan Neolux', 'Dongjin Semichem'],
            'equipment_suppliers': ['Wonik IPS', 'Koh Young']
        }
    },
    
    # Verified relationships (equity, revenue dependency, technology partnerships)
    'verified_relationships': {
        'shareholding': {
            ('Samsung Electronics', 'Wonik IPS'): {'stake': '7%', 'co_holder': 'Samsung Display'},
            ('Samsung Electronics', 'Semes'): {'relationship': 'Subsidiary'}
        },
        'revenue_dependency': {
            ('Nepes Ark', 'Samsung Electronics'): {'dependency': '85%+', 'products': 'PMIC testing'},
            ('Hana Micron', 'SK Hynix'): {'dependency': '60-70%', 'products': 'Memory packaging'},
            ('APTC', 'SK Hynix'): {'market_share': '90%', 'products': 'Metal etching equipment'}
        },
        'technology_partnerships': {
            ('Samsung Electronics', 'TSMC'): {'project': 'HBM4 joint development', 'participants': 'SK Hynix'},
            ('Dongjin Semichem', 'Samsung Electronics'): {'cooperation': 'Localization of Japan export restriction items', 'co_participant': 'Solvrain'}
        }
    },
    
    # 2020-2024 supply chain trends
    'supply_chain_trends': {
        'contract_evolution': 'Transition from annual bidding to 3-5 year long-term contracts',
        'localization': 'Domestic supply chain strengthening after Japan export restrictions',
        'customer_diversification': 'Customer diversification such as Wonik IPS expansion to SK Hynix',
        'global_expansion': 'US expansion of Solvrain and Dongjin Semichem following Samsung Electronics US factory'
    },
    
    # Data sources and update information
    'metadata': {
        'data_sources': 'Based on public corporate disclosures, news articles, and industry reports',
        'last_updated': 'June 2025',
        'verification_level': 'Public information verified'
    }
}