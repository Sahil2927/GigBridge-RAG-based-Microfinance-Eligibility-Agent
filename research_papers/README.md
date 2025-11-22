# Research Papers

Place your research PDF files in this folder.

The ingestion process will automatically process all `.pdf` files in this directory.

## Expected PDF Files

Based on the project requirements, you should have PDFs related to:
- Psychometric predictors of loan repayment
- Mobile phone usage and credit scoring
- Social capital and group lending
- Microfinance eligibility assessment
- Unconventional data for credit scoring

## File Naming

PDFs can have any name, but descriptive names are recommended:
- `psychometrics_loan_repayment.pdf`
- `mobile_metadata_credit_scoring.pdf`
- `social_capital_microfinance.pdf`
- etc.

## Processing

When you run the ingestion notebook or script, all PDFs in this folder will be:
1. Extracted for text content
2. Chunked into ~400-word segments
3. Embedded using sentence-transformers
4. Indexed in FAISS for retrieval


