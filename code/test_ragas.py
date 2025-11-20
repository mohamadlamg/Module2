import os
from typing import List, Dict
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Import your existing agent
from module2 import agent, State

load_dotenv()

# Configuration for Ragas with GROQ (instead of OpenAI)
evaluator_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Embeddings with Hugging Face (FREE and local)
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


def extract_contexts_from_agent_response(result: Dict) -> List[str]:
    """
    Extract contexts (sources) used by the agent
    """
    contexts = []
    
    for message in result.get('messages', []):
        # Retrieve ToolMessages that contain search results
        if hasattr(message, 'type') and message.type == 'tool':
            contexts.append(message.content)
    
    return contexts if contexts else ["No context retrieved"]


def run_agent_with_tracking(question: str) -> Dict:
    """
    Execute the agent and extract necessary info for Ragas
    """
    print(f"   Executing agent...")
    
    # Work directly with the question passed as parameter
    initial_state = State(messages=[{"role": "user", "content": question}])
    result = agent.invoke(initial_state)
    
    # Extract final answer
    answer = result['messages'][-1].content
    
    # Extract contexts (sources used)
    contexts = extract_contexts_from_agent_response(result)
    
    print(f" Response: {answer[:100]}...")
    print(f" Contexts found: {len(contexts)}")
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


# === TEST DATASET ===
test_data = [
    {
        "question": "What is quantum physics?",
        "ground_truth": "Quantum physics is a branch of physics that studies the behavior of matter and energy at the atomic and subatomic scale."
    },
    {
        "question": "What are the latest news on AI in Africa?",
        "ground_truth": "Artificial intelligence is developing rapidly in Africa with initiatives in several countries for education, health, and agriculture."
    },
    {
        "question": "Who discovered penicillin?",
        "ground_truth": "Penicillin was discovered by Alexander Fleming in 1928."
    },
    {
        "question": "What are the latest developments in nuclear fusion?",
        "ground_truth": "Nuclear fusion research is progressing with experiments like ITER and recent breakthroughs in inertial confinement fusion."
    },
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed."
    }
]


def evaluate_agent():
    """
    Evaluate agent performance with Ragas
    """
    print("\n" + "="*60)
    print(" STARTING RAGAS EVALUATION")
    print("="*60)
    print(f" Evaluator LLM: Groq (llama-3.1-8b-instant)")
    print(f" Embeddings: HuggingFace (all-MiniLM-L6-v2)")
    print(f" Number of tests: {len(test_data)}")
    print("="*60 + "\n")
    
    # Collect results
    evaluation_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for i, test_case in enumerate(test_data, 1):
        print(f" Test {i}/{len(test_data)}")
        print(f" Question: {test_case['question']}")
        
        try:
            # Execute agent with test_case question
            result = run_agent_with_tracking(test_case["question"])
            
            # Collect data
            evaluation_data["question"].append(result["question"])
            evaluation_data["answer"].append(result["answer"])
            evaluation_data["contexts"].append(result["contexts"])
            evaluation_data["ground_truth"].append(test_case["ground_truth"])
            
            print(f"Test successful\n")
            
        except Exception as e:
            print(f" Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    if not evaluation_data["question"]:
        print("No test succeeded. Check your agent.")
        return None
    
    # Create dataset for Ragas
    print("\n Creating evaluation dataset...")
    dataset = Dataset.from_dict(evaluation_data)
    
    print(f"Dataset created with {len(dataset)} examples\n")
    
    print("="*60)
    print("RAGAS EVALUATION IN PROGRESS...")
    print("This may take a few minutes...")
    print("="*60 + "\n")
    
    # Evaluate with Ragas metrics
    try:
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False
        )
        return result
        
    except Exception as e:
        print(f"\n Error during Ragas evaluation: {e}")
        print("\nTroubleshooting tips:")
        print("  - Check that GROQ_API_KEY is configured")
        print("  - Try with fewer tests")
        import traceback
        traceback.print_exc()
        return None


def display_results(result):
    """
    Display results in a readable format
    """
    if result is None:
        print(" No results to display")
        return None
    
    print("\n" + "="*60)
    print("RAGAS EVALUATION RESULTS")
    print("="*60 + "\n")
    
    # Global scores
    df = result.to_pandas()
    
    print("AVERAGE SCORES:")
    print("-" * 40)
    
    metrics_info = {
        'faithfulness': 'Faithfulness (no hallucinations)',
        'answer_relevancy': ' Answer Relevancy',
        'context_recall': ' Context Recall'
    }
    
    for metric in metrics_info.items():
        if metric in df.columns:
            score = df[metric].mean()
            if score >= 0.8:
                status = "EXCELLENT"
            elif score >= 0.6:
                status = "GOOD"
            else:
                status = "NEEDS IMPROVEMENT"
            
            print(f"   Score: {score:.3f} ({status})")
    
    print("\n" + "="*60)
    
    # Overall score
    available_metrics = [col for col in ['faithfulness', 'answer_relevancy', 'context_recall'] if col in df.columns]
    if available_metrics:
        overall_score = df[available_metrics].mean().mean()
        print(f"\n OVERALL SCORE: {overall_score:.3f}/1.000")
        
        if overall_score >= 0.8:
            print("\n Excellent! Your agent performs very well!")
            print("   Production quality")
        elif overall_score >= 0.6:
            print("\nGood! There's room for improvement.")
            print("    Some adjustments recommended")
        else:
            print("\nWarning! The agent needs optimization.")
            print("   Revision needed")
    
    print("\n" + "="*60)
    
    return df



# === EXECUTION ===
if __name__ == "__main__":
    print("\n RAGAS EVALUATION - Multi-Tool Research Agent")
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        print(" Error: GROQ_API_KEY missing")
        print(" Add it to your .env file")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY missing")
        print("   Some tests might fail")
    
    try:
        # 1. Run evaluation
        results = evaluate_agent()
        
        # 2. Display results
        if results is not None:
            df = display_results(results)
            
        else:
            print("\n Evaluation could not be completed")
            print("Check the logs above to identify the issue")
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n Fatal error: {e}")
        import traceback
        traceback.print_exc()