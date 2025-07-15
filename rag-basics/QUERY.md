This document explains the complete workflow of how your RAG system processes a literature query,
  referencing the specific implementation in your local test repository.

  Overview: The Complete Pipeline

  When you ask a question like "What is simulation-based inference?", here's the full journey from
  query to response:

  User Question → Query Embedding → Vector Search → Context Assembly → LLM Generation → Final Response

  Detailed Step-by-Step Workflow

  Step 1: Query Embedding Creation

  Location: step2_simple_rag.py:query() method, lines ~200-210

  # Create embedding for the question
  response = self.client.embeddings.create(
      input=[question],
      model=OpenAI_Embedding_Model  # "text-embedding-3-large"
  )
  query_embedding = response.data[0].embedding

  What happens:
  - Your question text is sent to OpenAI's embedding model
  - The model converts your question into a 3072-dimensional vector
  - This vector represents the semantic meaning of your question
  - Cost: ~$0.00013 per 1K tokens (very cheap for questions)

  Step 2: Vector Similarity Search

  Location: step2_simple_rag.py:query() method, lines ~215-225

  # Search ChromaDB
  results = self.collection.query(
      query_embeddings=[query_embedding],
      n_results=n_results  # Default: 5 chunks
  )

  What happens:
  - ChromaDB compares your query embedding with all 104 stored document embeddings
  - Uses cosine similarity to find the most semantically similar chunks
  - Returns the top N most relevant document chunks
  - Speed: Milliseconds (very fast vector math)

  The Math Behind It:
  # Simplified version of what ChromaDB does:
  similarities = []
  for doc_embedding in stored_embeddings:
      similarity = cosine_similarity(query_embedding, doc_embedding)
      similarities.append(similarity)

  # Return top N most similar
  top_chunks = sorted(similarities, reverse=True)[:n_results]

  Step 3: Context Assembly

  Location: step2_simple_rag.py:query() method, lines ~230-240

  # Format context from retrieved documents
  context = ""
  sources = []

  for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
      context += f"\n\n--- Source {i+1}: {metadata['source']} ---\n{doc}"
      sources.append(metadata['source'])

  What happens:
  - Takes the retrieved chunks and assembles them into a context string
  - Adds source attribution for each chunk
  - Collects unique source filenames for citation
  - Creates a structured input for the LLM

  Example assembled context:
  --- Source 1: 2507.03086v1.md ---
  Simulation-based inference (SBI) has become an important tool in cosmology...

  --- Source 2: 2502.08416v2.md ---
  SBI utilizes forward simulations to generate synthetic data...

  Step 4: LLM Response Generation

  Location: step2_simple_rag.py:query() method, lines ~245-270

  prompt = f"""Based on the following context from research papers, answer the question.
  Be specific and cite the sources when possible.

  Context:
  {context}

  Question: {question}

  Answer:"""

  chat_response = self.client.chat.completions.create(
      model=OPENAI_GEN_MODEL,  # "gpt-4o-mini"
      messages=[
          {"role": "system", "content": "You are a helpful research assistant..."},
          {"role": "user", "content": prompt}
      ],
      temperature=TEMPERATURE,  # 0.1 for consistency
      max_tokens=500
  )

  What happens:
  - Constructs a structured prompt with retrieved context
  - Sends to OpenAI's chat completion API
  - LLM generates an answer based on the provided context
  - Cost: ~$0.000150 input + $0.000600 output per 1K tokens

  Step 5: Response Assembly and Return

  Location: step2_simple_rag.py:query() method, lines ~275-285

  answer = chat_response.choices[0].message.content

  result = {
      'question': question,
      'answer': answer,
      'sources': list(set(sources)),  # Remove duplicates
      'context_snippets': results['documents'][0]
  }

  return result

  What happens:
  - Extracts the generated answer
  - Packages it with metadata (sources, original chunks)
  - Returns structured response to user

  Configuration Parameters That Affect Each Step

  Query Embedding (Step 1)

  # local_config.py
  OpenAI_Embedding_Model = "text-embedding-3-large"  # Model quality vs cost

  Vector Search (Step 2)

  # local_config.py
  TOP_K = 15                    # Number of chunks to retrieve
  DISTANCE_THRESHOLD = 0.4      # Minimum similarity score
  n_results = 5                 # Default in query() method

  Context Assembly (Step 3)

  # local_config.py
  CHUNK_SIZE = 3000            # Size of each retrievable chunk
  CHUNK_OVERLAP = 300          # Overlap between chunks for context

  LLM Generation (Step 4)

  # local_config.py
  OPENAI_GEN_MODEL = "gpt-4o-mini"  # Model selection
  TEMPERATURE = 0.1                  # Response consistency
  max_tokens = 500                   # Response length limit

  Performance Characteristics

  Typical Query Timeline:
  1. Query Embedding: ~100-200ms (network + OpenAI processing)
  2. Vector Search: ~5-10ms (local ChromaDB computation)
  3. Context Assembly: ~1-2ms (string operations)
  4. LLM Generation: ~1-3 seconds (network + OpenAI processing)
  5. Response Assembly: ~1ms (data packaging)

  Total: ~1.5-3.5 seconds per query

  Cost Breakdown per Query:
  - Query embedding: ~$0.00001 (for typical question length)
  - Vector search: $0 (local computation)
  - LLM generation: ~$0.001-0.003 (depends on context length and response)

  Debugging and Monitoring

  Check retrieval quality:
  # In step2_simple_rag.py, add after line ~225:
  print(f"Retrieved {len(results['documents'][0])} chunks")
  for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
      print(f"Chunk {i+1}: {meta['source']} - {doc[:100]}...")

  Monitor similarity scores:
  # Add to see how well chunks match your query
  if 'distances' in results:
      print(f"Similarity scores: {results['distances'][0]}")

  Check context length:
  # Before LLM call, add:
  print(f"Context length: {len(context)} characters")
  print(f"Estimated tokens: ~{len(context.split())}")

  Common Issues and Solutions

  Poor Retrieval Results:
  - Lower DISTANCE_THRESHOLD to get more chunks
  - Increase TOP_K to see more candidates
  - Check if documents were chunked properly

  Expensive Queries:
  - Reduce max_tokens in generation
  - Use smaller embedding model (text-embedding-3-small)
  - Reduce TOP_K to retrieve fewer chunks

  Slow Performance:
  - Check internet connection (both embedding and generation use OpenAI API)
  - Reduce n_results parameter
  - Consider caching common queries

  This workflow ensures that your questions are matched against the most semantically relevant parts of
   your literature corpus, providing accurate and well-sourced answers while maintaining cost
  efficiency and good performance.