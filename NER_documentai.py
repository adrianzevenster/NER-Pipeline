#!/usr/bin/env python3

import pandas as pd
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json
import os

@dataclass
class EntityAnalysis:
    entity_type: str
    confidence: float
    frequency: int
    document_coverage: int
    examples: List[str]
    patterns: List[str]
    priority_score: float

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
                    r'\b0[7-9]\d{9}\b',  # Nigerian mobile
                    r'\b01\d{7,8}\b',    # Nigerian landline
                    r'\b\+234[7-9]\d{9}\b',  # International format
                    r'\b\d{10,11}\b',    # General phone pattern
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
                    r'\b\d{11}\b',  # NIN format
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

    def load_tokens_data(self, csv_file: str) -> pd.DataFrame:
        """Load the kyc_tokens.csv file."""
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} tokens from {df['file_uri'].nunique()} documents")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {csv_file}. Please ensure the file exists.")

    def extract_document_type(self, file_uri: str) -> str:
        """Extract document type from file URI."""
        file_uri_upper = file_uri.upper()

        # Define document type patterns
        type_patterns = {
            'WATER_BILL': ['WATER_BILL', 'WATER'],
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

        for doc_type, patterns in type_patterns.items():
            if any(pattern in file_uri_upper for pattern in patterns):
                return doc_type

        return 'UNKNOWN'

    def classify_token(self, token: str, doc_type: str, context_tokens: List[str] = None) -> Dict[str, float]:
        """Classify a token against all entity patterns."""
        if not token or len(token.strip()) < 1:
            return {'OTHER': 0.1}

        token = str(token).strip()
        classifications = {}

        entity_patterns = {**self.common_patterns, **self.document_type_patterns.get(doc_type, {})}
        for entity_type, config in entity_patterns.items():
            score = 0.0

            # Check regex patterns
            pattern_match = False
            for pattern in config['patterns']:
                try:
                    if re.search(pattern, token, re.IGNORECASE):
                        pattern_match = True
                        score += 0.7
                        break
                except re.error:
                    continue

            # Check keyword matches
            keyword_match = False
            for keyword in config['keywords']:
                if keyword.lower() in token.lower():
                    keyword_match = True
                    score += 0.3
                    break

            # Context bonus
            if context_tokens:
                context_text = ' '.join(context_tokens).lower()
                for keyword in config['keywords']:
                    if keyword.lower() in context_text:
                        score += 0.2
                        break

            # Apply confidence weight
            final_score = min(score * config['confidence_weight'], 1.0)

            if final_score > 0:
                classifications[entity_type] = final_score

        # Return highest scoring classification or OTHER
        if classifications:
            best_entity = max(classifications.items(), key=lambda x: x[1])
            return {best_entity[0]: best_entity[1]}
        else:
            return {'OTHER': 0.1}

    def analyze_document(self, doc_tokens: pd.DataFrame, doc_type: str, file_uri: str) -> Dict[str, List[Dict]]:
        """Analyze tokens from a single document."""
        entities = defaultdict(list)

        # Sort tokens by position for context
        doc_tokens = doc_tokens.sort_values(['page', 'ymin', 'xmin']).reset_index(drop=True)

        for idx, row in doc_tokens.iterrows():
            token = str(row['token']).strip() if pd.notna(row['token']) else ''

            if not token or len(token) < 1:
                continue

            confidence = float(row['confidence']) if pd.notna(row['confidence']) else 0.0

            # Get context tokens (nearby tokens)
            context_start = max(0, idx - 3)
            context_end = min(len(doc_tokens), idx + 4)
            context_tokens = []

            for context_idx in range(context_start, context_end):
                if context_idx != idx:
                    context_token = str(doc_tokens.iloc[context_idx]['token']).strip()
                    if context_token:
                        context_tokens.append(context_token)

            # Classify token using document-specific patterns
            classification = self.classify_token(token, doc_type, context_tokens)

            for entity_type, entity_confidence in classification.items():
                entities[entity_type].append({
                    'token': token,
                    'classification_confidence': entity_confidence,
                    'ocr_confidence': confidence,
                    'combined_confidence': entity_confidence * confidence,
                    'position': {
                        'page': row.get('page', 1),
                        'xmin': row.get('xmin', 0),
                        'ymin': row.get('ymin', 0),
                        'xmax': row.get('xmax', 0),
                        'ymax': row.get('ymax', 0)
                    },
                    'file_uri': file_uri
                })

        return entities

    def analyze_all_documents(self, tokens_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze all documents and group by document type."""
        results = {
            'document_types': defaultdict(lambda: {
                'count': 0,
                'entities': defaultdict(list),
                'total_tokens': 0
            }),
            'overall_stats': defaultdict(list),
            'document_analyses': []
        }

        # Group by document
        for file_uri in tokens_df['file_uri'].unique():
            doc_tokens = tokens_df[tokens_df['file_uri'] == file_uri]
            doc_type = self.extract_document_type(file_uri)

            # Analyze document
            doc_entities = self.analyze_document(doc_tokens, doc_type, file_uri)

            # Store document analysis
            doc_analysis = {
                'file_uri': file_uri,
                'document_type': doc_type,
                'total_tokens': len(doc_tokens),
                'entities': doc_entities
            }
            results['document_analyses'].append(doc_analysis)

            # Update document type stats
            results['document_types'][doc_type]['count'] += 1
            results['document_types'][doc_type]['total_tokens'] += len(doc_tokens)

            # Aggregate entities by document type
            for entity_type, entity_list in doc_entities.items():
                results['document_types'][doc_type]['entities'][entity_type].extend(entity_list)
                results['overall_stats'][entity_type].extend(entity_list)

        return results

    def calculate_entity_priority(self, entity_data: List[Dict]) -> float:
        """Calculate priority score for an entity type."""
        if not entity_data:
            return 0.0

        # Factors for priority calculation
        frequency = len(entity_data)
        avg_confidence = np.mean([item['combined_confidence'] for item in entity_data])
        unique_tokens = len(set(item['token'] for item in entity_data))

        # Normalize frequency (log scale to prevent dominance)
        frequency_score = min(np.log(frequency + 1) / 10, 1.0)
        confidence_score = avg_confidence
        diversity_score = min(unique_tokens / max(frequency, 1), 1.0)

        # Weighted combination
        priority = (frequency_score * 0.3 + confidence_score * 0.5 + diversity_score * 0.2)

        return priority

    def generate_entity_analysis(self, results: Dict) -> List[EntityAnalysis]:
        """Generate comprehensive entity analysis."""
        entity_analyses = []

        for entity_type, entity_data in results['overall_stats'].items():
            if not entity_data:
                continue

            # Calculate statistics
            frequency = len(entity_data)
            avg_confidence = np.mean([item['combined_confidence'] for item in entity_data])
            unique_tokens = list(set(item['token'] for item in entity_data))

            # Count document coverage
            documents_with_entity = set()
            for doc_analysis in results['document_analyses']:
                if entity_type in doc_analysis['entities'] and doc_analysis['entities'][entity_type]:
                    documents_with_entity.add(doc_analysis['file_uri'])

            document_coverage = len(documents_with_entity)

            # Get top examples (by confidence)
            sorted_examples = sorted(entity_data, key=lambda x: x['combined_confidence'], reverse=True)
            examples = [item['token'] for item in sorted_examples[:10]]
            examples = list(dict.fromkeys(examples))[:5]  # Remove duplicates, keep top 5

            # Get patterns for this entity type (aggregate from common + all doc types)
            patterns = []
            if entity_type in self.common_patterns:
                patterns.extend(self.common_patterns[entity_type].get('patterns', []))
            for doc_patterns in self.document_type_patterns.values():
                if entity_type in doc_patterns:
                    patterns.extend(doc_patterns[entity_type].get('patterns', []))
            patterns = list(set(patterns))  # Unique

            # Calculate priority score
            priority_score = self.calculate_entity_priority(entity_data)

            analysis = EntityAnalysis(
                entity_type=entity_type,
                confidence=avg_confidence,
                frequency=frequency,
                document_coverage=document_coverage,
                examples=examples,
                patterns=patterns,
                priority_score=priority_score
            )

            entity_analyses.append(analysis)

        # Sort by priority score
        entity_analyses.sort(key=lambda x: x.priority_score, reverse=True)

        return entity_analyses

    def generate_report(self, tokens_df: pd.DataFrame, output_file: str = "entity_analysis_report.txt"):
        """Generate comprehensive analysis report organized by document type."""
        print("Analyzing tokens by document type...")
        results = self.analyze_all_documents(tokens_df)

        print("Generating document-type-specific entity analysis...")

        # Generate report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DOCUMENT-TYPE-SPECIFIC ENTITY ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Summary
            f.write("SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Documents: {len(results['document_analyses'])}\n")
            f.write(f"Total Tokens: {sum(doc['total_tokens'] for doc in results['document_analyses'])}\n")
            f.write(f"Document Types: {len(results['document_types'])}\n\n")

            # Document type breakdown with specific entities
            for doc_type, stats in results['document_types'].items():
                f.write(f"\nDOCUMENT TYPE: {doc_type}\n")
                f.write("=" * 40 + "\n")
                f.write(f"Documents: {stats['count']}\n")
                f.write(f"Total Tokens: {stats['total_tokens']}\n")
                f.write(f"Avg Tokens per Document: {stats['total_tokens'] // stats['count']}\n\n")

                # Generate entity analysis for this document type
                doc_entity_analyses = self.generate_entity_analysis_for_document_type(
                    stats['entities'], doc_type
                )

                if doc_entity_analyses:
                    f.write("RECOMMENDED ENTITIES (BY PRIORITY):\n")
                    f.write("-" * 35 + "\n")

                    for i, analysis in enumerate(doc_entity_analyses, 1):
                        f.write(f"{i}. {analysis.entity_type}\n")
                        f.write(f"   Priority Score: {analysis.priority_score:.3f}\n")
                        f.write(f"   Confidence: {analysis.confidence:.3f}\n")
                        f.write(f"   Frequency: {analysis.frequency}\n")
                        f.write(f"   Document Coverage: {analysis.document_coverage}/{stats['count']}\n")
                        f.write(f"   Examples: {', '.join(analysis.examples[:3])}\n")
                        if len(analysis.examples) > 3:
                            f.write(f"   ... and {len(analysis.examples) - 3} more\n")
                        f.write(f"   Regex Patterns: {len(analysis.patterns)}\n\n")

                    # Implementation recommendations for this document type
                    high_priority = [a for a in doc_entity_analyses if a.priority_score > 0.7]
                    medium_priority = [a for a in doc_entity_analyses if 0.4 < a.priority_score <= 0.7]
                    low_priority = [a for a in doc_entity_analyses if a.priority_score <= 0.4]

                    f.write("IMPLEMENTATION RECOMMENDATIONS:\n")
                    f.write("-" * 35 + "\n")

                    if high_priority:
                        f.write("PHASE 1 (High Priority):\n")
                        for analysis in high_priority:
                            f.write(f"  - {analysis.entity_type} (score: {analysis.priority_score:.3f})\n")
                        f.write("\n")

                    if medium_priority:
                        f.write("PHASE 2 (Medium Priority):\n")
                        for analysis in medium_priority:
                            f.write(f"  - {analysis.entity_type} (score: {analysis.priority_score:.3f})\n")
                        f.write("\n")

                    if low_priority:
                        f.write("PHASE 3 (Low Priority):\n")
                        for analysis in low_priority:
                            f.write(f"  - {analysis.entity_type} (score: {analysis.priority_score:.3f})\n")
                        f.write("\n")
                else:
                    f.write("No significant entities detected for this document type.\n\n")

                f.write("-" * 40 + "\n")

        print(f"Document-type-specific report saved to: {output_file}")

        # Generate summary statistics
        return self._generate_summary_statistics(results)

    def generate_entity_analysis_for_document_type(self, entity_data: Dict, doc_type: str) -> List[EntityAnalysis]:
        """Generate entity analysis for a specific document type."""
        entity_analyses = []

        for entity_type, entity_list in entity_data.items():
            if not entity_list or entity_type == 'OTHER':
                continue

            # Calculate statistics
            frequency = len(entity_list)
            if frequency == 0:
                continue

            avg_confidence = np.mean([item['combined_confidence'] for item in entity_list])
            unique_tokens = list(set(item['token'] for item in entity_list))

            # Document coverage based on unique file_uris
            files_with_entity = set(item['file_uri'] for item in entity_list if 'file_uri' in item)
            document_coverage = len(files_with_entity)

            # Get top examples (by confidence)
            sorted_examples = sorted(entity_list, key=lambda x: x['combined_confidence'], reverse=True)
            examples = [item['token'] for item in sorted_examples[:15]]
            examples = list(dict.fromkeys(examples))[:10]  # Remove duplicates, keep top 10

            # Get patterns for this entity type and document type
            patterns = []
            doc_patterns = self.document_type_patterns.get(doc_type, {})
            if entity_type in doc_patterns:
                patterns = doc_patterns[entity_type].get('patterns', [])
            elif entity_type in self.common_patterns:
                patterns = self.common_patterns[entity_type].get('patterns', [])

            # Calculate priority score
            priority_score = self.calculate_entity_priority(entity_list)

            analysis = EntityAnalysis(
                entity_type=entity_type,
                confidence=avg_confidence,
                frequency=frequency,
                document_coverage=document_coverage,
                examples=examples,
                patterns=patterns,
                priority_score=priority_score
            )

            entity_analyses.append(analysis)

        # Sort by priority score
        entity_analyses.sort(key=lambda x: x.priority_score, reverse=True)

        return entity_analyses

    def _generate_summary_statistics(self, results: Dict) -> Dict:
        """Generate overall summary statistics."""
        summary = {
            'document_types': {},
            'overall_recommendations': {
                'high_priority_by_doc_type': {},
                'medium_priority_by_doc_type': {},
                'low_priority_by_doc_type': {}
            }
        }

        for doc_type, stats in results['document_types'].items():
            doc_entity_analyses = self.generate_entity_analysis_for_document_type(
                stats['entities'], doc_type
            )

            summary['document_types'][doc_type] = {
                'total_entities': len(doc_entity_analyses),
                'high_priority': [a.entity_type for a in doc_entity_analyses if a.priority_score > 0.5],
                'medium_priority': [a.entity_type for a in doc_entity_analyses if 0.2 < a.priority_score <= 0.5],
                'low_priority': [a.entity_type for a in doc_entity_analyses if a.priority_score <= 0.2],
                'recommended_labels': [a.entity_type for a in doc_entity_analyses if a.priority_score > 0.2]
            }

            summary['overall_recommendations']['high_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['high_priority']
            summary['overall_recommendations']['medium_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['medium_priority']
            summary['overall_recommendations']['low_priority_by_doc_type'][doc_type] = summary['document_types'][doc_type]['low_priority']

        return summary

    def export_ml_config_by_document_type(self, summary_stats: Dict, output_file: str = "ml_entity_config_by_document_type.json"):
        """Export ML configuration organized by document type."""
        config = {
            'document_types': {},
            'global_patterns': self.common_patterns
        }

        for doc_type, stats in summary_stats['document_types'].items():
            if stats['recommended_labels']:
                doc_patterns = self.document_type_patterns.get(doc_type, {})

                config['document_types'][doc_type] = {
                    'entity_labels': stats['recommended_labels'],
                    'high_priority': stats['high_priority'],
                    'medium_priority': stats['medium_priority'],
                    'low_priority': stats['low_priority'],
                    'patterns': {
                        entity: doc_patterns.get(entity, {}).get('patterns', [])
                        for entity in stats['recommended_labels']
                    },
                    'confidence_thresholds': {
                        entity: doc_patterns.get(entity, {}).get('confidence_weight', 0.3)
                        for entity in stats['recommended_labels']
                    }
                }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"Document-type-specific ML configuration saved to: {output_file}")
        return config

    def _get_essential_entities_for_doc_type(self, doc_type: str) -> List[str]:
        """Get essential entities that should be present for each document type."""
        essential_entities = {
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
        return essential_entities.get(doc_type, [])

    def export_ml_config(self, entity_analyses: List[EntityAnalysis], output_file: str = "ml_entity_config.json"):
        """Export configuration for ML model training."""
        config = {
            'entity_labels': [analysis.entity_type for analysis in entity_analyses],
            'confidence_thresholds': {
                analysis.entity_type: max(0.5, analysis.confidence)
                for analysis in entity_analyses
            },
            'implementation_phases': {
                'phase1': [a.entity_type for a in entity_analyses if a.priority_score > 0.7],
                'phase2': [a.entity_type for a in entity_analyses if 0.3 < a.priority_score <= 0.7],
                'phase3': [a.entity_type for a in entity_analyses if a.priority_score <= 0.3]
            },
            'patterns': {
                analysis.entity_type: analysis.patterns
                for analysis in entity_analyses if analysis.patterns
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"ML configuration saved to: {output_file}")
        return config

def main():
    """Main function to run document-type-specific analysis."""
    # Initialize analyzer
    analyzer = EntityLabelAnalyzer()

    # Load tokens data
    try:
        tokens_csv = os.environ.get("TOKENS_CSV", "kyc_tokens_documentai.csv")
        tokens_df = analyzer.load_tokens_data(tokens_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Generate comprehensive document-type-specific report
    print("Running document-type-specific analysis...")
    summary_stats = analyzer.generate_report(tokens_df, "document_type_entity_analysis.txt")

    # Export ML configuration by document type
    ml_config = analyzer.export_ml_config_by_document_type(summary_stats)

    # Save detailed results to CSV organized by document type
    results_data = []
    for doc_type, stats in summary_stats['document_types'].items():
        for entity_type in stats['recommended_labels']:
            priority = 'HIGH' if entity_type in stats['high_priority'] else \
                'MEDIUM' if entity_type in stats['medium_priority'] else 'LOW'

            # Get pattern info
            doc_patterns = analyzer.document_type_patterns.get(doc_type, {})
            pattern_count = len(doc_patterns.get(entity_type, {}).get('patterns', []))
            confidence_weight = doc_patterns.get(entity_type, {}).get('confidence_weight', 0.5)

            results_data.append({
                'document_type': doc_type,
                'entity_type': entity_type,
                'priority': priority,
                'confidence_weight': confidence_weight,
                'pattern_count': pattern_count,
                'recommended': True
            })

    results_df = pd.DataFrame(results_data)
    results_df.to_csv("entity_analysis_by_document_type.csv", index=False)
    print("Detailed results saved to: entity_analysis_by_document_type.csv")

    # Print summary by document type
    print(f"\nDOCUMENT-TYPE-SPECIFIC ANALYSIS COMPLETE")
    print("=" * 50)

    for doc_type, stats in summary_stats['document_types'].items():
        print(f"\n{doc_type}:")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  High priority: {len(stats['high_priority'])}")
        print(f"  Recommended labels: {stats['recommended_labels']}")

        if stats['high_priority']:
            print(f"  Priority entities: {', '.join(stats['high_priority'])}")

    print(f"\nAll document types covered:")
    for doc_type in summary_stats['document_types'].keys():
        print(f"  - {doc_type}")

if __name__ == "__main__":
    main()