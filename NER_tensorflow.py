#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entity suggestion & report generation using TensorFlow/keras-ocr tokens as the base.

Inputs
------
- kyc_tokens_tensorflow.csv  (default; override with --tokens)
  Expected columns: file_uri, page, token, xmin, xmax, ymin, ymax, confidence (optional), doc_type (optional)

Outputs
-------
- document_type_entity_analysis.txt
- ml_entity_config_by_document_type.json
- entity_analysis_by_document_type.csv
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os

import numpy as np
import pandas as pd

# Keep TensorFlow as the "base" dependency (no model inference here; analysis only)
import tensorflow as tf  # noqa: F401


# ========================= Data Model =========================

@dataclass
class EntityAnalysis:
    entity_type: str
    confidence: float
    frequency: int
    document_coverage: int
    examples: List[str]
    patterns: List[str]
    priority_score: float


# ========================= Analyzer ===========================

class EntityLabelAnalyzer:
    def __init__(self):
        # Document type-specific entity patterns for Nigerian documents
        self.document_type_patterns = {
            'BANK_STATEMENT': {
                'ACCOUNT_NUMBER': {
                    'patterns': [r'\b\d{10,12}\b', r'(?i)(?:account|acc|a/c)[\s\-#:]*(\d{8,})'],
                    'keywords': ['account', 'acc', 'a/c', 'number'],
                    'confidence_weight': 0.9
                },
                'ROUTING_NUMBER': {
                    'patterns': [r'\b\d{9}\b', r'(?i)(?:routing|sort)[\s\-#:]*(\d{9})'],
                    'keywords': ['routing', 'sort', 'code'],
                    'confidence_weight': 0.9
                },
                'TRANSACTION_AMOUNT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*', r'\b\d{1,3}(,\d{3})*\.\d{2}\b'],
                    'keywords': ['amount', 'debit', 'credit', 'balance', 'withdrawal', 'deposit'],
                    'confidence_weight': 0.85
                },
                'TRANSACTION_DATE': {
                    'patterns': [r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'],
                    'keywords': ['date', 'transaction', 'posted'],
                    'confidence_weight': 0.9
                },
                'BANK_NAME': {
                    'patterns': [r'(?i)\b(?:first|zenith|gtb|uba|access|fidelity|union|sterling).*bank\b'],
                    'keywords': ['bank', 'plc', 'limited'],
                    'confidence_weight': 0.8
                }
            },

            'DRIVERS_LICENSE': {
                'LICENSE_NUMBER': {
                    'patterns': [r'\b[A-Z]{3}\d{9}[A-Z]{2}\b', r'\b[A-Z0-9]{8,15}\b'],
                    'keywords': ['license', 'licence', 'dl', 'number'],
                    'confidence_weight': 0.9
                },
                'FULL_NAME': {
                    'patterns': [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'],
                    'keywords': ['name', 'full name', 'surname', 'first name'],
                    'confidence_weight': 0.8
                },
                'DATE_OF_BIRTH': {
                    'patterns': [r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'],
                    'keywords': ['dob', 'birth', 'date of birth', 'born'],
                    'confidence_weight': 0.9
                },
                'ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['address', 'residence', 'street', 'road'],
                    'confidence_weight': 0.7
                },
                'EXPIRY_DATE': {
                    'patterns': [r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'],
                    'keywords': ['expires', 'expiry', 'valid until', 'exp'],
                    'confidence_weight': 0.9
                },
                'SEX': {
                    'patterns': [r'\b[MF]\b', r'(?i)\b(?:male|female)\b'],
                    'keywords': ['sex', 'gender'],
                    'confidence_weight': 0.95
                }
            },

            'VOTERS_CARD': {
                'VIN_NUMBER': {
                    'patterns': [r'\b\d{19}\b', r'\b[A-Z0-9]{19}\b'],
                    'keywords': ['vin', 'voter', 'identification', 'number'],
                    'confidence_weight': 0.95
                },
                'PU_CODE': {
                    'patterns': [r'\b\d{2}/\d{2}/\d{2}/\d{3}\b', r'\b[A-Z]{2}/\d{2}/\d{2}/\d{3}\b'],
                    'keywords': ['pu', 'polling unit', 'code'],
                    'confidence_weight': 0.9
                },
                'FULL_NAME': {
                    'patterns': [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'],
                    'keywords': ['name', 'surname', 'first name'],
                    'confidence_weight': 0.8
                },
                'STATE': {
                    'patterns': [r'(?i)\b(?:lagos|kano|rivers|kaduna|oyo|imo|delta|anambra|akwa ibom|cross river)\b'],
                    'keywords': ['state', 'origin'],
                    'confidence_weight': 0.85
                },
                'LGA': {
                    'patterns': [r'(?i)\b[a-z]+\s+(?:local government|lga)\b'],
                    'keywords': ['lga', 'local government', 'area'],
                    'confidence_weight': 0.8
                }
            },

            'ELECTRICITY_BILL': {
                'METER_NUMBER': {
                    'patterns': [r'\b\d{8,15}\b', r'(?i)(?:meter|mtr)[\s\-#:]*(\d{8,})'],
                    'keywords': ['meter', 'mtr', 'number'],
                    'confidence_weight': 0.9
                },
                'CUSTOMER_ID': {
                    'patterns': [r'\b\d{8,12}\b'],
                    'keywords': ['customer', 'account', 'id'],
                    'confidence_weight': 0.85
                },
                'BILL_AMOUNT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*', r'\b\d{1,3}(,\d{3})*\.\d{2}\b'],
                    'keywords': ['amount', 'total', 'due', 'balance', 'bill'],
                    'confidence_weight': 0.9
                },
                'KWH_USAGE': {
                    'patterns': [r'\b\d+\.?\d*\s*(?:kwh|kw)\b'],
                    'keywords': ['usage', 'consumption', 'kwh', 'units'],
                    'confidence_weight': 0.9
                },
                'SERVICE_ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['service address', 'address', 'location'],
                    'confidence_weight': 0.75
                },
                'TARIFF_CLASS': {
                    'patterns': [r'(?i)\b(?:residential|commercial|industrial|r1|r2|c1|c2)\b'],
                    'keywords': ['tariff', 'class', 'residential', 'commercial'],
                    'confidence_weight': 0.85
                }
            },

            'WATER_BILL': {
                'CUSTOMER_NUMBER': {
                    'patterns': [r'\b\d{8,15}\b'],
                    'keywords': ['customer', 'account', 'number'],
                    'confidence_weight': 0.9
                },
                'BILL_AMOUNT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*', r'\b\d{1,3}(,\d{3})*\.\d{2}\b'],
                    'keywords': ['amount', 'total', 'due', 'balance'],
                    'confidence_weight': 0.9
                },
                'METER_READING': {
                    'patterns': [r'\b\d{1,6}\s*(?:m³|cubic|litres)\b', r'\b(?:current|previous):\s*\d+\b'],
                    'keywords': ['reading', 'current', 'previous', 'consumption'],
                    'confidence_weight': 0.85
                },
                'SERVICE_ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['service address', 'address', 'property'],
                    'confidence_weight': 0.75
                },
                'WATER_BOARD_NAME': {
                    'patterns': [r'(?i)\b(?:lagos|kano|rivers|kaduna)\s+(?:water|state)\s+(?:board|corporation)\b'],
                    'keywords': ['water', 'board', 'corporation', 'authority'],
                    'confidence_weight': 0.8
                }
            },

            'WASTE_MANAGEMENT_BILL': {
                'PROPERTY_ID': {
                    'patterns': [r'\b[A-Z0-9]{8,15}\b', r'\b\d{8,12}\b'],
                    'keywords': ['property', 'id', 'code', 'number'],
                    'confidence_weight': 0.85
                },
                'BILL_AMOUNT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*'],
                    'keywords': ['amount', 'fee', 'charge', 'due'],
                    'confidence_weight': 0.9
                },
                'PROPERTY_ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['property address', 'address', 'location'],
                    'confidence_weight': 0.75
                },
                'WASTE_AUTHORITY': {
                    'patterns': [r'(?i)\b(?:lawma|waste|management|authority|agency)\b'],
                    'keywords': ['lawma', 'waste', 'management', 'authority'],
                    'confidence_weight': 0.8
                },
                'ASSESSMENT_PERIOD': {
                    'patterns': [r'\b\d{4}\b', r'(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'],
                    'keywords': ['assessment', 'period', 'year'],
                    'confidence_weight': 0.8
                }
            },

            'SOLICITORS_LETTER': {
                'CASE_NUMBER': {
                    'patterns': [r'\b[A-Z]{1,3}/\d{3,4}/\d{4}\b', r'\bsuit\s+no[\.:]?\s*[A-Z0-9/]+\b'],
                    'keywords': ['case', 'suit', 'matter', 'ref', 'number'],
                    'confidence_weight': 0.9
                },
                'LAW_FIRM_NAME': {
                    'patterns': [r'(?i)\b[A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)*\s+(?:chambers|associates|partners)\b'],
                    'keywords': ['chambers', 'associates', 'partners', 'solicitors'],
                    'confidence_weight': 0.8
                },
                'CLIENT_NAME': {
                    'patterns': [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'],
                    'keywords': ['client', 'applicant', 'plaintiff', 'defendant'],
                    'confidence_weight': 0.75
                },
                'COURT_NAME': {
                    'patterns': [r'(?i)\b(?:high court|magistrate|federal high court|court of appeal)\b'],
                    'keywords': ['court', 'tribunal', 'high court'],
                    'confidence_weight': 0.85
                },
                'LEGAL_FEE': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*'],
                    'keywords': ['fee', 'cost', 'professional fee', 'amount'],
                    'confidence_weight': 0.8
                }
            },

            'TENANCY_AGREEMENT': {
                'TENANT_NAME': {
                    'patterns': [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'],
                    'keywords': ['tenant', 'lessee', 'occupant'],
                    'confidence_weight': 0.8
                },
                'LANDLORD_NAME': {
                    'patterns': [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'],
                    'keywords': ['landlord', 'lessor', 'owner'],
                    'confidence_weight': 0.8
                },
                'PROPERTY_ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['property', 'premises', 'address', 'location'],
                    'confidence_weight': 0.85
                },
                'MONTHLY_RENT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*'],
                    'keywords': ['rent', 'monthly', 'rental', 'amount'],
                    'confidence_weight': 0.9
                },
                'LEASE_DURATION': {
                    'patterns': [r'\b\d{1,2}\s+(?:years?|months?)\b'],
                    'keywords': ['duration', 'term', 'period', 'lease'],
                    'confidence_weight': 0.85
                },
                'COMMENCEMENT_DATE': {
                    'patterns': [r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'],
                    'keywords': ['commencement', 'start', 'beginning', 'effective'],
                    'confidence_weight': 0.9
                }
            },

            'GENERIC_UTILITY_BILL': {
                'ACCOUNT_NUMBER': {
                    'patterns': [r'\b\d{8,15}\b'],
                    'keywords': ['account', 'customer', 'number'],
                    'confidence_weight': 0.85
                },
                'BILL_AMOUNT': {
                    'patterns': [r'[\$£€¥₦N]\s*[\d,]+\.?\d*'],
                    'keywords': ['amount', 'total', 'due', 'balance'],
                    'confidence_weight': 0.9
                },
                'DUE_DATE': {
                    'patterns': [r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'],
                    'keywords': ['due', 'payment due', 'deadline'],
                    'confidence_weight': 0.9
                },
                'SERVICE_PROVIDER': {
                    'patterns': [r'(?i)\b(?:telecommunications|telecom|cable|internet|gas|utility)\b.*'],
                    'keywords': ['provider', 'company', 'services', 'utility'],
                    'confidence_weight': 0.8
                },
                'SERVICE_ADDRESS': {
                    'patterns': [r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:street|road|avenue)\b'],
                    'keywords': ['service address', 'billing address', 'location'],
                    'confidence_weight': 0.75
                }
            }
        }

        # Common patterns that might appear across document types
        self.common_patterns = {
            'DATE': {
                'patterns': [
                    r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
                    r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
                    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                ],
                'keywords': ['date', 'issued', 'expires', 'due'],
                'confidence_weight': 0.95
            },
            'PHONE_NUMBER': {
                'patterns': [
                    r'\b0[7-9]\d{9}\b',      # Nigerian mobile
                    r'\b01\d{7,8}\b',        # Nigerian landline
                    r'\b\+234[7-9]\d{9}\b',  # Intl format
                    r'\b\d{10,11}\b',        # General phone
                ],
                'keywords': ['phone', 'tel', 'mobile', 'contact', 'call', 'gsm'],
                'confidence_weight': 0.8
            },
            'EMAIL': {
                'patterns': [r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'],
                'keywords': ['email', 'mail', 'e-mail'],
                'confidence_weight': 0.95
            },
            'REFERENCE_NUMBER': {
                'patterns': [
                    r'\b\d{10,20}\b',
                    r'\b[A-Z]{2,4}\d{8,15}\b',
                    r'(?i)(?:ref|reference|transaction|txn|id)[\s\-#:]*([A-Z0-9]{6,})',
                ],
                'keywords': ['reference', 'ref', 'transaction', 'txn', 'id', 'number', 'account'],
                'confidence_weight': 0.85
            },
            'CUSTOMER_NAME': {
                'patterns': [
                    r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b',
                    r'\b[A-Z\s]{4,}\b',
                    r'(?i)(?:mr|mrs|miss|dr|prof|alhaji|chief)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+',
                ],
                'keywords': ['name', 'customer', 'payer', 'surname', 'given', 'full name', 'mr', 'mrs', 'alhaji'],
                'confidence_weight': 0.7
            },
            'ADDRESS': {
                'patterns': [
                    r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Close|Crescent)\b',
                    r'(?i)(?:address|addr)[\s\-:]*(\d+\s+[A-Za-z\s]+)',
                    r'\b[A-Z][a-z]+\s+(?:State|Island|Area|Estate|Plaza)\b',
                ],
                'keywords': ['address', 'addr', 'street', 'road', 'avenue', 'close', 'estate', 'state', 'location'],
                'confidence_weight': 0.75
            },
            'ORGANIZATION': {
                'patterns': [
                    r'(?i)\b(?:bank|electric|water|waste|management|authority|board|company|corporation|service|ministry|revenue)\b.*',
                    r'(?i).*\b(?:bank|electric|water|waste|management|authority|board|company|corporation|service|ministry|revenue)\b',
                ],
                'keywords': ['bank', 'electric', 'water', 'waste', 'management', 'authority', 'board', 'company', 'ministry', 'revenue', 'service'],
                'confidence_weight': 0.85
            },
            'ID_NUMBER': {
                'patterns': [
                    r'\b\d{11}\b',  # NIN
                    r'\b[A-Z0-9]{10,15}\b',
                    r'(?i)(?:nin|id|identification)[\s\-#:]*([A-Z0-9]{8,})',
                ],
                'keywords': ['nin', 'id', 'identification', 'number', 'national'],
                'confidence_weight': 0.9
            },
            'ACCOUNT_TYPE': {
                'patterns': [
                    r'(?i)\b(?:domestic|commercial|industrial|residential|business|savings|current)\b',
                ],
                'keywords': ['domestic', 'commercial', 'industrial', 'residential', 'business', 'savings', 'current', 'type'],
                'confidence_weight': 0.9
            },
            'BILLING_PERIOD': {
                'patterns': [
                    r'(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*\d{4}\b',
                    r'(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4}\b',
                ],
                'keywords': ['billing', 'period', 'month', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'],
                'confidence_weight': 0.8
            }
        }

    # ---------------- I/O ----------------

    def load_tokens_data(self, csv_file: str) -> pd.DataFrame:
        """Load tokens CSV (TensorFlow/keras-ocr output)."""
        df = pd.read_csv(csv_file)
        # normalize column presence
        if "token" not in df.columns:
            raise ValueError("CSV must contain a 'token' column.")
        if "file_uri" not in df.columns:
            raise ValueError("CSV must contain a 'file_uri' column.")
        # geometry defaults
        for col in ["page", "xmin", "xmax", "ymin", "ymax"]:
            if col not in df.columns:
                df[col] = 0
        # confidence normalize -> float in [0,1]
        if "confidence" not in df.columns:
            df["confidence"] = 0.9
        else:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
            df.loc[df["confidence"] > 1.0, "confidence"] = df["confidence"] / 100.0
            df["confidence"] = df["confidence"].clip(0.0, 1.0)

        # doc_type: use provided or infer from file_uri
        if "doc_type" not in df.columns:
            df["doc_type"] = df["file_uri"].astype(str).apply(self.extract_document_type)

        print(f"Loaded {len(df)} tokens from {df['file_uri'].nunique()} documents "
              f"across {df['doc_type'].nunique()} document types.")
        return df

    # ------------- Utilities -------------

    def extract_document_type(self, file_uri: str) -> str:
        """Extract document type from file URI (fallback when doc_type absent)."""
        file_uri_upper = (file_uri or "").upper()
        type_patterns = {
            'WATER_BILL': ['WATER_BILL', 'WATER', 'UTILITY'],
            'BANK_STATEMENT': ['BANK_STATEMENT', 'BANK', 'STATEMENT'],
            'DRIVERS_LICENSE': ['DRIVER_LICENSE', 'DRIVERS_LICENSE', 'LICENSE', 'DL'],
            'ID_CARD': ['ID_CARD', 'NATIONAL_ID', 'NIN'],
            'ELECTRICITY_BILL': ['ELECTRICITY_BILL', 'ELECTRIC', 'POWER'],
            'RECEIPT': ['RECEIPT', 'PAYMENT', 'INVOICE'],
            'VOTERS_CARD': ['VOTERS_CARD', 'VOTER', 'VIN'],
            'WASTE_MANAGEMENT_BILL': ['WASTE_MANAGEMENT_BILL', 'WASTE', 'LAWMA'],
            'SOLICITORS_LETTER': ['SOLICITORS_LETTER', 'SOLICITOR', 'LETTER', 'LEGAL'],
            'TENANCY_AGREEMENT': ['TENANCY_AGREEMENT', 'TENANCY', 'LEASE', 'AGREEMENT'],
            'GENERIC_UTILITY_BILL': ['GENERIC_UTILITY_BILL', 'UTILITY_BILL']
        }
        for doc_type, pats in type_patterns.items():
            if any(p in file_uri_upper for p in pats):
                return doc_type
        return 'UNKNOWN'

    def classify_token(self, token: str, doc_type: str, context_tokens: Optional[List[str]] = None) -> Dict[str, float]:
        """Classify a token against entity patterns (common + doc-type specific)."""
        token = (token or "").strip()
        if not token:
            return {'OTHER': 0.0}

        classifications: Dict[str, float] = {}
        entity_patterns = {**self.common_patterns, **self.document_type_patterns.get(doc_type, {})}

        context_text = " ".join(context_tokens or []).lower()
        for entity_type, cfg in entity_patterns.items():
            score = 0.0
            # regex
            pattern_hit = False
            for pat in cfg.get('patterns', []):
                try:
                    if re.search(pat, token, flags=re.IGNORECASE):
                        score += 0.7
                        pattern_hit = True
                        break
                except re.error:
                    continue
            # keyword on token
            for kw in cfg.get('keywords', []):
                if kw.lower() in token.lower():
                    score += 0.3
                    break
            # context bonus
            if context_text:
                for kw in cfg.get('keywords', []):
                    if kw.lower() in context_text:
                        score += 0.2
                        break
            if score > 0:
                score = min(score * float(cfg.get('confidence_weight', 0.8)), 1.0)
                classifications[entity_type] = score

        if classifications:
            best = max(classifications.items(), key=lambda kv: kv[1])
            return {best[0]: best[1]}
        return {'OTHER': 0.0}

    # ------------- Per-document analysis -------------

    def analyze_document(self, doc_tokens: pd.DataFrame, doc_type: str, file_uri: str) -> Dict[str, List[Dict]]:
        """Analyze one document’s tokens -> entity buckets with confidences & positions."""
        entities: Dict[str, List[Dict]] = defaultdict(list)
        doc_tokens = doc_tokens.sort_values(['page', 'ymin', 'xmin']).reset_index(drop=True)

        for i, row in doc_tokens.iterrows():
            tok = str(row.get('token', '') or '').strip()
            if not tok:
                continue
            ocr_conf = float(row.get('confidence', 0.0))
            # local context +/- 3 tokens
            a, b = max(0, i - 3), min(len(doc_tokens), i + 4)
            ctx = [str(doc_tokens.iloc[k]['token']).strip()
                   for k in range(a, b) if k != i and pd.notna(doc_tokens.iloc[k]['token'])]

            cls = self.classify_token(tok, doc_type, ctx)
            for ent, cls_conf in cls.items():
                if ent == 'OTHER' or cls_conf <= 0:
                    continue
                entities[ent].append({
                    'token': tok,
                    'classification_confidence': float(cls_conf),
                    'ocr_confidence': float(ocr_conf),
                    'combined_confidence': float(cls_conf * ocr_conf),
                    'position': {
                        'page': int(row.get('page', 1) or 1),
                        'xmin': float(row.get('xmin', 0.0) or 0.0),
                        'ymin': float(row.get('ymin', 0.0) or 0.0),
                        'xmax': float(row.get('xmax', 0.0) or 0.0),
                        'ymax': float(row.get('ymax', 0.0) or 0.0),
                    },
                    'file_uri': file_uri
                })
        return entities

    def analyze_all_documents(self, tokens_df: pd.DataFrame) -> Dict[str, Dict]:
        """Run analysis across all files, aggregate by document type and globally."""
        results = {
            'document_types': defaultdict(lambda: {'count': 0, 'entities': defaultdict(list), 'total_tokens': 0}),
            'overall_stats': defaultdict(list),
            'document_analyses': []
        }

        for file_uri, g in tokens_df.groupby('file_uri'):
            doc_type = str(g['doc_type'].iloc[0])
            ent_map = self.analyze_document(g, doc_type, file_uri)

            # per-doc store
            results['document_analyses'].append({
                'file_uri': file_uri,
                'document_type': doc_type,
                'total_tokens': len(g),
                'entities': ent_map
            })

            # type totals
            results['document_types'][doc_type]['count'] += 1
            results['document_types'][doc_type]['total_tokens'] += len(g)

            # aggregate per type & global
            for ent, items in ent_map.items():
                results['document_types'][doc_type]['entities'][ent].extend(items)
                results['overall_stats'][ent].extend(items)

        return results

    # ------------- Scoring & reporting -------------

    def calculate_entity_priority(self, items: List[Dict]) -> float:
        """Priority score blends frequency, average combined confidence, diversity."""
        if not items:
            return 0.0
        freq = len(items)
        avg_conf = float(np.mean([x['combined_confidence'] for x in items])) if items else 0.0
        uniq_tokens = len({x['token'] for x in items})
        freq_score = min(np.log(freq + 1) / 10.0, 1.0)
        conf_score = avg_conf  # already 0..1
        diversity = min(uniq_tokens / max(freq, 1), 1.0)
        return float(0.3 * freq_score + 0.5 * conf_score + 0.2 * diversity)

    def generate_entity_analysis(self, results: Dict) -> List[EntityAnalysis]:
        """Global entity analysis across all document types."""
        analyses: List[EntityAnalysis] = []
        for ent, items in results['overall_stats'].items():
            if not items:
                continue
            freq = len(items)
            avg_conf = float(np.mean([x['combined_confidence'] for x in items])) if items else 0.0

            # document coverage
            docs = {x['file_uri'] for x in items if 'file_uri' in x}
            coverage = len(docs)

            # examples (top by combined confidence, de-duped)
            examples = []
            for x in sorted(items, key=lambda r: r['combined_confidence'], reverse=True):
                t = x['token']
                if t not in examples:
                    examples.append(t)
                if len(examples) >= 5:
                    break

            # collect patterns for this entity if present
            pats = []
            if ent in self.common_patterns:
                pats += self.common_patterns[ent].get('patterns', [])
            for dmap in self.document_type_patterns.values():
                if ent in dmap:
                    pats += dmap[ent].get('patterns', [])
            pats = list(dict.fromkeys(pats))  # unique

            analyses.append(EntityAnalysis(
                entity_type=ent,
                confidence=avg_conf,
                frequency=freq,
                document_coverage=coverage,
                examples=examples,
                patterns=pats,
                priority_score=self.calculate_entity_priority(items),
            ))
        analyses.sort(key=lambda a: a.priority_score, reverse=True)
        return analyses

    def generate_entity_analysis_for_document_type(self, ent_map: Dict[str, List[Dict]], doc_type: str) -> List[EntityAnalysis]:
        """Entity analysis for a specific doc type using its entity buckets."""
        out: List[EntityAnalysis] = []
        for ent, items in ent_map.items():
            if ent == 'OTHER' or not items:
                continue
            freq = len(items)
            avg_conf = float(np.mean([x['combined_confidence'] for x in items])) if items else 0.0
            docs = {x['file_uri'] for x in items if 'file_uri' in x}
            coverage = len(docs)

            # top examples
            examples = []
            for x in sorted(items, key=lambda r: r['combined_confidence'], reverse=True):
                t = x['token']
                if t not in examples:
                    examples.append(t)
                if len(examples) >= 10:
                    break

            # patterns for this ent + doc_type
            pats = []
            dmap = self.document_type_patterns.get(doc_type, {})
            if ent in dmap:
                pats = dmap[ent].get('patterns', [])
            elif ent in self.common_patterns:
                pats = self.common_patterns[ent].get('patterns', [])

            out.append(EntityAnalysis(
                entity_type=ent,
                confidence=avg_conf,
                frequency=freq,
                document_coverage=coverage,
                examples=examples,
                patterns=pats,
                priority_score=self.calculate_entity_priority(items),
            ))
        out.sort(key=lambda a: a.priority_score, reverse=True)
        return out

    def _get_essential_entities_for_doc_type(self, doc_type: str) -> List[str]:
        """(Optional) Essential entities per doc type."""
        essentials = {
            'VOTERS_CARD': ['VIN_NUMBER', 'FULL_NAME', 'STATE'],
            'ELECTRICITY_BILL': ['METER_NUMBER', 'BILL_AMOUNT', 'CUSTOMER_ID'],
            'WATER_BILL': ['CUSTOMER_NUMBER', 'BILL_AMOUNT'],
            'BANK_STATEMENT': ['ACCOUNT_NUMBER', 'TRANSACTION_AMOUNT', 'BANK_NAME'],
            'DRIVERS_LICENSE': ['LICENSE_NUMBER', 'FULL_NAME', 'DATE_OF_BIRTH'],
            'WASTE_MANAGEMENT_BILL': ['PROPERTY_ID', 'BILL_AMOUNT'],
            'SOLICITORS_LETTER': ['LAW_FIRM_NAME', 'CASE_NUMBER'],
            'TENANCY_AGREEMENT': ['TENANT_NAME', 'LANDLORD_NAME', 'MONTHLY_RENT'],
            'GENERIC_UTILITY_BILL': ['ACCOUNT_NUMBER', 'BILL_AMOUNT', 'DUE_DATE']
        }
        return essentials.get(doc_type, [])

    # ------------- Reporting & export -------------

    def generate_report(self, tokens_df: pd.DataFrame, output_file: str = "document_type_entity_analysis.txt"):
        """Write a human-readable report per document type and return summary stats."""
        results = self.analyze_all_documents(tokens_df)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DOCUMENT-TYPE-SPECIFIC ENTITY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Documents: {len(results['document_analyses'])}\n")
            f.write(f"Total Tokens: {sum(d['total_tokens'] for d in results['document_analyses'])}\n")
            f.write(f"Document Types: {len(results['document_types'])}\n\n")

            for doc_type, stats in results['document_types'].items():
                f.write(f"\nDOCUMENT TYPE: {doc_type}\n")
                f.write("=" * 40 + "\n")
                f.write(f"Documents: {stats['count']}\n")
                f.write(f"Total Tokens: {stats['total_tokens']}\n")
                avg_tok = stats['total_tokens'] // max(stats['count'], 1)
                f.write(f"Avg Tokens per Document: {avg_tok}\n\n")

                analyses = self.generate_entity_analysis_for_document_type(stats['entities'], doc_type)
                if analyses:
                    f.write("RECOMMENDED ENTITIES (BY PRIORITY):\n")
                    f.write("-" * 35 + "\n")
                    for i, a in enumerate(analyses, 1):
                        f.write(f"{i}. {a.entity_type}\n")
                        f.write(f"   Priority Score: {a.priority_score:.3f}\n")
                        f.write(f"   Confidence: {a.confidence:.3f}\n")
                        f.write(f"   Frequency: {a.frequency}\n")
                        f.write(f"   Document Coverage: {a.document_coverage}/{stats['count']}\n")
                        if a.examples:
                            extras = f" ... and {max(len(a.examples)-3,0)} more" if len(a.examples) > 3 else ""
                            f.write(f"   Examples: {', '.join(a.examples[:3])}{extras}\n")
                        f.write(f"   Regex Patterns: {len(a.patterns)}\n\n")

                    high = [x for x in analyses if x.priority_score > 0.7]
                    mid  = [x for x in analyses if 0.4 < x.priority_score <= 0.7]
                    low  = [x for x in analyses if x.priority_score <= 0.4]

                    f.write("IMPLEMENTATION RECOMMENDATIONS:\n")
                    f.write("-" * 35 + "\n")
                    if high:
                        f.write("PHASE 1 (High Priority):\n")
                        for a in high: f.write(f"  - {a.entity_type} (score: {a.priority_score:.3f})\n")
                        f.write("\n")
                    if mid:
                        f.write("PHASE 2 (Medium Priority):\n")
                        for a in mid: f.write(f"  - {a.entity_type} (score: {a.priority_score:.3f})\n")
                        f.write("\n")
                    if low:
                        f.write("PHASE 3 (Low Priority):\n")
                        for a in low: f.write(f"  - {a.entity_type} (score: {a.priority_score:.3f})\n")
                        f.write("\n")
                else:
                    f.write("No significant entities detected for this document type.\n\n")
                f.write("-" * 40 + "\n")

        print(f"Document-type-specific report saved to: {output_file}")
        return self._generate_summary_statistics(results)

    def _generate_summary_statistics(self, results: Dict) -> Dict:
        """Overall per-type recommendations to feed config exporters."""
        summary = {'document_types': {}, 'overall_recommendations': {
            'high_priority_by_doc_type': {}, 'medium_priority_by_doc_type': {}, 'low_priority_by_doc_type': {}
        }}

        for doc_type, stats in results['document_types'].items():
            analyses = self.generate_entity_analysis_for_document_type(stats['entities'], doc_type)

            summary['document_types'][doc_type] = {
                'total_entities': len(analyses),
                'high_priority': [a.entity_type for a in analyses if a.priority_score > 0.5],
                'medium_priority': [a.entity_type for a in analyses if 0.2 < a.priority_score <= 0.5],
                'low_priority': [a.entity_type for a in analyses if a.priority_score <= 0.2],
                'recommended_labels': [a.entity_type for a in analyses if a.priority_score > 0.2]
            }
            summary['overall_recommendations']['high_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['high_priority']
            summary['overall_recommendations']['medium_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['medium_priority']
            summary['overall_recommendations']['low_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['low_priority']

        return summary

    def export_ml_config_by_document_type(self, summary_stats: Dict, output_file: str = "ml_entity_config_by_document_type.json"):
        """Emit JSON config of recommended entities & thresholds per doc type."""
        config = {'document_types': {}, 'global_patterns': self.common_patterns}
        for doc_type, stats in summary_stats['document_types'].items():
            if stats['recommended_labels']:
                doc_pats = self.document_type_patterns.get(doc_type, {})
                config['document_types'][doc_type] = {
                    'entity_labels': stats['recommended_labels'],
                    'high_priority': stats['high_priority'],
                    'medium_priority': stats['medium_priority'],
                    'low_priority': stats['low_priority'],
                    'patterns': {
                        ent: doc_pats.get(ent, {}).get('patterns', [])
                        for ent in stats['recommended_labels']
                    },
                    'confidence_thresholds': {
                        ent: doc_pats.get(ent, {}).get('confidence_weight', 0.3)
                        for ent in stats['recommended_labels']
                    }
                }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Document-type-specific ML configuration saved to: {output_file}")
        return config

    def export_ml_config(self, entity_analyses: List[EntityAnalysis], output_file: str = "ml_entity_config.json"):
        """(Optional) Global config variant if you want a cross-type set."""
        config = {
            'entity_labels': [a.entity_type for a in entity_analyses],
            'confidence_thresholds': {a.entity_type: max(0.5, a.confidence) for a in entity_analyses},
            'implementation_phases': {
                'phase1': [a.entity_type for a in entity_analyses if a.priority_score > 0.7],
                'phase2': [a.entity_type for a in entity_analyses if 0.3 < a.priority_score <= 0.7],
                'phase3': [a.entity_type for a in entity_analyses if a.priority_score <= 0.3],
            },
            'patterns': {a.entity_type: a.patterns for a in entity_analyses if a.patterns}
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"Global ML configuration saved to: {output_file}")
        return config


# ========================= CLI ===============================

# --- replace your parse/main entry with this ---

import sys

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entity analysis using TensorFlow token CSV as base.")
    p.add_argument("--tokens", default="kyc_tokens_tensorflow.csv",
                   help="Path to TensorFlow/keras-ocr tokens CSV (default: kyc_tokens_tensorflow.csv)")
    p.add_argument("--report", default="document_type_entity_analysis_tensorflow.txt",
                   help="Output path for the human-readable report")
    p.add_argument("--out-config", default="ml_entity_config_by_document_type_tensorflow.json",
                   help="Output JSON with per-type ML config")
    p.add_argument("--out-csv", default="entity_analysis_by_document_type_tensorflow.csv",
                   help="CSV summary per document type")
    return p.parse_args(argv)

    args.tokens = os.environ.get('TOKENS_CSV', args.tokens)
    args.report = os.environ.get('REPORT_FILE', args.report)
    args.out_config = os.environ.get('OUT_CONFIG', args.out_config)
    args.out_csv = os.environ.get('OUT_CSV', args.out_csv)

def main(argv=None):
    args = parse_args(argv)
    analyzer = EntityLabelAnalyzer()
    try:
        df = analyzer.load_tokens_data(args.tokens)
    except Exception as e:
        print(f"ERROR loading tokens: {e}")
        return
    summary_stats = analyzer.generate_report(df, args.report)
    analyzer.export_ml_config_by_document_type(summary_stats, args.out_config)

    rows = []
    for doc_type, stats in summary_stats['document_types'].items():
        doc_pats = analyzer.document_type_patterns.get(doc_type, {})
        for ent in stats['recommended_labels']:
            priority = ('HIGH' if ent in stats['high_priority'] else
                        'MEDIUM' if ent in stats['medium_priority'] else 'LOW')
            rows.append({
                'document_type': doc_type,
                'entity_type': ent,
                'priority': priority,
                'confidence_weight': doc_pats.get(ent, {}).get('confidence_weight', 0.5),
                'pattern_count': len(doc_pats.get(ent, {}).get('patterns', [])),
                'recommended': True
            })
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Detailed results saved to: {args.out_csv}")

    print("\nDOCUMENT-TYPE-SPECIFIC ANALYSIS COMPLETE")
    print("=" * 50)
    for doc_type, stats in summary_stats['document_types'].items():
        print(f"\n{doc_type}:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  High priority: {len(stats['high_priority'])}")
        print(f"  Recommended labels: {stats['recommended_labels']}")
        if stats['high_priority']:
            print(f"  Priority entities: {', '.join(stats['high_priority'])}")
    print("\nAll document types covered:")
    for doc_type in summary_stats['document_types'].keys():
        print(f"  - {doc_type}")

if __name__ == "__main__":
    # If running inside IPython/Jupyter, strip foreign args
    if "ipykernel" in sys.modules:
        main(argv=[])
    else:
        main()

