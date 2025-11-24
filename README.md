# AI Price Prediction with RAG Pipeline

This project implements an **AI-powered price prediction system** that uses a combination of:

- A **vector database** (ChromaDB) to find similar products
- **Retrieval-Augmented Generation (RAG)** to provide rich context
- A **Large Language Model (LLM)** (Claude / Hugging Face models) to reason about prices

The core idea:  
> Instead of predicting price from numbers alone, we let an LLM **see similar products and their prices**, then **infer** a reasonable price for a new item and explain its reasoning.

---

## Main File

- `redoing_205_week8_day2_Claude_predict_price_RAG_pipeline.ipynb`  
  The end-to-end notebook containing:
  - Environment setup
  - Data loading & inspection
  - Vector database creation (ChromaDB)
  - Retrieval logic
  - LLM prompt construction
  - Price prediction + explanations

---

## Technologies Used

- **Python**
- **PyTorch** & **Transformers** – for embeddings / models
- **ChromaDB** – vector database for semantic search
- **Claude API** (or another LLM) – to generate price predictions
- **Hugging Face Hub** – to access models or embeddings
- (Optionally) **Google Colab + Google Drive** – to run and store data

---

## High-Level Architecture

1. **Data**: Product information (title, description, attributes, price).
2. **Embedding**: Convert product descriptions into dense vectors.
3. **Vector Store**: Store vectors in **ChromaDB**.
4. **Retrieval**: For a new product, retrieve the most similar existing products.
5. **RAG Prompt**: Build a prompt that includes:
   - The new product’s attributes
   - The retrieved similar products + their prices
6. **LLM Call**: Ask the LLM to:
   - Propose a price
   - Explain the reasoning based on the context

---

## What the Notebook Does

### PIP Installations

Installs the core libraries needed:

- `torch` and `transformers` for model & embedding support  
- `chromadb` for the vector database  
- Any extra libraries like `Items`, data utilities, etc.

> This ensures the environment (typically Colab) has all dependencies.

---

### GPU and System Checks

- Detects whether a **GPU** is available.
- Prints information about the device and versions.
- Helps confirm that the notebook can run heavy models efficiently.

> If no GPU is found, the notebook may still work but slower.

---

### Setup Claude API Client

- Configures the **Claude (Anthropic) API** client.
- Loads API keys from environment variables or configuration.
- Sets up helper functions to send prompts and receive responses.

> This is how the notebook communicates with the LLM for price prediction.

---

### Login to Hugging Face

- Authenticates with **Hugging Face Hub** (if required).
- Allows the notebook to:
  - Download embedding models
  - Access private models or datasets (if configured).

---

### Mount Google Drive (Optional / Colab)

- Mounts **Google Drive** to the Colab runtime.
- Lets you:
  - Load datasets stored in Drive
  - Save results / checkpoints back to Drive.

---

### Main Imports

- Imports all required Python modules:
  - `torch`, `transformers`
  - `chromadb`
  - `json`, `os`, etc.
- Prints a success message when everything is imported correctly.

> At this point, the environment is fully prepared.

---

### Helper Functions

Defines utility functions used throughout the pipeline, such as:

- `description(item)` – builds a clean textual description from an item dict  
  (e.g., combining title, brand, category, attributes, etc.)
- Parsing and formatting functions
- Functions for logging or pretty-printing retrieved items
- Small helpers to construct prompts and handle responses

> These helpers keep the notebook readable and the logic reusable.

---

### Copy Dataset to Colab VM

- Copies or loads the **train** and **test** datasets into the runtime.
- Typically uses:
  - Local files
  - Google Drive
  - A provided dataset path

> After this step, the notebook can access `train` and `test` product lists.

---

### Checking Train Data

The notebook inspects the training data:

- Loops through the first few items in the **train** list.
- Prints each item’s:
  - Raw dictionary
  - Main fields and attribute names
- Helps you understand:
  - What each product object looks like
  - Which fields are useful (e.g., title, description, price, attributes)

> This step is about **data exploration** and debugging.

---

### Checking Test Data

Similarly, for the **test** dataset:

- Prints a few test items.
- Lets you confirm that test items have:
  - The same structure as train items
  - The necessary fields for prediction (except price, which may be unknown)

> This ensures the model can be applied consistently to both train and test.

---

### ChromaDB: Initializing and Resetting a Vector Collection

- Connects to **ChromaDB** and creates a new collection (e.g., `products`).
- Resets the collection if it already exists (to avoid duplicates).
- Defines:
  - Collection name
  - Metadata schema (if needed)

> This is where product embeddings will be stored for similarity search.

---

### Building and Storing Embeddings


1. Iterates over each **train** product.
2. Uses an embedding model (e.g., from `transformers`) to:
   - Convert the product `description(item)` into a vector.
3. Stores in ChromaDB:
   - `id`: product id
   - `embedding`: dense vector
   - `metadata`: product fields (title, price, etc.)

> This turns your dataset into a searchable **semantic index**.

---

### Similarity Search (Finding Similar Products)

When you have a **target product** (e.g., from the test set):

1. Build its **query text** (again with `description(item)`).
2. Embed this query into a vector.
3. Ask ChromaDB for the **top-k most similar products**.
4. Retrieve:
   - Their descriptions
   - Their prices
   - Any relevant attributes

> These similar products become **context** that will guide the LLM’s prediction.

---

### Constructing the RAG Prompt

The notebook then constructs a prompt to send to the LLM, typically containing:

- A clear **instruction**, e.g.:  
  “You are an expert pricing analyst. Estimate a reasonable price for this product based on similar items.”
- The **target product** details:
  - Title, category, brand, attributes
- The **retrieved similar products**:
  - Each with title + attributes + known price


  
