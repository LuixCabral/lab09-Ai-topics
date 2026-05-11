# Lab 09 - RAG Avançado

Implementação de um pipeline RAG (Retrieval-Augmented Generation) com suporte a HyDE, indexação HNSW e re-ranking com Cross-Encoders.

## 🛠️ Como Executar

1. **Configurar Ambiente:**
   Crie um ambiente virtual e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar API Key:**
   Crie um arquivo `.env` na raiz do projeto com sua chave do Gemini:
   ```env
   GEMINI_API_KEY=sua_chave_aqui
   ```
   *Nota: Sem a chave, o sistema utilizará um texto de fallback estático para o HyDE.*

3. **Executar:**
   ```bash
   python main.py
   ```

## 📂 Estrutura do Projeto

- `main.py`: Orquestrador do pipeline (caching, loop de busca e re-ranking).
- `components/data.py`: Base de dados simulada.
- `components/hyde.py`: Integração com Gemini para geração do documento hipotético.
- `components/indexing.py`: Gerenciamento do índice (build, save e load usando FAISS).

## 🧠 Tarefa Analítica (Vector Indexing)

- **FAISS HNSW:** Utilizamos o índice HNSWFlat do FAISS para busca aproximada de vizinhos mais próximos.
- **Normalização:** Os vetores são normalizados (L2) para que a métrica de Produto Interno (Inner Product) equivalha à Similaridade de Cosseno.
- **Persistência:** O uso de `index.faiss` demonstra como sistemas reais lidam com coleções estáticas ou incrementais para reduzir latência de inicialização.
