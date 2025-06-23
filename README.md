# CTD-RAG-System

## Project Structure

```
CTD-RAG-System/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── courserag/                   # Main package
│   ├── __init__.py
│   ├── core/                    # Core RAG components
│   │   ├── __init__.py
│   │   ├── rag_system.py        
│   │   ├── document_loader.py
│   │   ├── vector_store.py
│   │   └── rag_chain.py
│   ├── config/                  # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py          
│   │   └── logging_config.py
│   ├── utils/                   # Utilities and helpers
│   │   ├── __init__.py
│   │   ├── cache.py             
│   │   └── helpers.py          
│   └── processing/              # Data processing
│       ├── __init__.py
│       ├── papers.py            
│       ├── presentations.py    
│       ├── announcements.py     
│       └── transcripts.py      
│
├── web/                         # Web interface
│   ├── __init__.py
│   ├── app.py                   
│   ├── static/                  
│   │   └── styles.css
│   └── components/              
│       └── __init__.py
│
├── cli/                         # Command-line tools
│   ├── __init__.py
│   ├── populate_db.py           
│   └── benchmark.py             
│
├── data/                        # Data directories
│   ├── raw/                     
│   ├── clean/               
│   └── cache/                   
│
├── logs/                        # Log files
└── .github/                     
```



## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AchrafYndz/CTD-RAG-System
   cd CTD-RAG-System
   ```

2. **Install dependencies using [uv](https://docs.astral.sh/uv/):**
   ```bash
   uv sync
   ```

3. **Set up environment variables:**

   - Copy the example environment file:
     ```bash
     cp example.env .env
     ```
   - Edit `.env` and set your `OPENAI_API_KEY`

## Usage

### Running the Streamlit App

To start the Streamlit application, navigate to the project root directory and run the following command:
```bash
streamlit run web/app.py
```
