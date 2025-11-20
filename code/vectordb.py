import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser


class VectorDB:
    def __init__(self,collection_name,embedding_model_name):
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name 

        self.client = chromadb.PersistentClient(path='rag_clients_db')


        self.embedding_model = HuggingFaceEmbeddings(
            model_name = embedding_model_name
            
        )
        

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"Vector base initialized with success ! : {self.collection}")

    def chunk_text(self,text,chunk_size=1000):
        
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        
        chunks = text_splitter.split_text(text)
        return chunks
        
       
    
    def add_documents(self,documents):

       
        if not documents:
            print("Not document found")
            return
        print(f"Processing {len(documents)} documents")

        all_chunks = []
        chunk_metadata = []

        for doc_idx, document in enumerate(documents):
            chunks = self.chunk_text(document)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk)
                })
            print(f"Chunks size : {len(all_chunks)}")

        # Générer les embeddings par batch
        batch_size = 1000
        embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            print(f"Generating embeddings for batch {i // batch_size + 1} "
                f"({len(batch)} chunks)...")
            batch_embeddings = self.embedding_model.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        # Générer les IDs uniques
        next_id = self.collection.count()
        ids = [f"chunk_{next_id + i}" for i in range(len(all_chunks))]

        # Ajouter par batch
        for i in range(0, len(all_chunks), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_docs = all_chunks[i:i + batch_size]
            batch_meta = chunk_metadata[i:i + batch_size]

            print(f"Adding batch {i // batch_size + 1} to collection "
                f"({len(batch_docs)} chunks)...")
            self.collection.add(
                embeddings=batch_embeddings,
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )

    def search(self,query,n_results:int=5) -> dict:
        """Searching user queries in database"""

        if not query :
            raise ValueError("Query empty cannot find anything")
        
        if n_results < 0 :
            raise ValueError("n_results must be positive..")
        
        query = query.strip()
        
        query_vector = self.embedding_model.embed_query(query)

        outputs = self.collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
        )

        
        return{
            "documents": outputs["documents"][0] if outputs["documents"] else [],
            "metadatas": outputs["metadatas"][0] if outputs["metadatas"] else [],
            "distances": outputs["distances"][0] if outputs["distances"] else [],
            "ids": outputs["ids"][0] if outputs["ids"] else []
        }
           
            
           
    


