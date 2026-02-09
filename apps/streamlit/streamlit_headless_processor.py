#!/usr/bin/env python3
"""
Headless version of the Streamlit cluster QA app.
Runs the same RAG functionality without the web interface.

Usage:
    python streamlit_headless_processor.py questions.txt output.txt [options]
    
Options:
    --cluster-id CLUSTER_ID     Process questions for specific cluster (default: all)
    --model MODEL_NAME          Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
    --pipeline PIPELINE         Pipeline type: ultra_fast or standard (default: ultra_fast)
    --format FORMAT            Response format: detailed, bullet_points, summary, comparative, technical (default: detailed)
    --max-tokens TOKENS        Maximum tokens for response (default: 4000)
    --top-k K                  Number of top documents to retrieve (default: 5)
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Suppress warnings before importing the heavy modules
import os
import warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import the functions from cluster_qa_app without triggering Streamlit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock streamlit to avoid errors when importing cluster_qa_app functions
class MockStreamlit:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            pass
        return mock_func
    
    @property
    def session_state(self):
        return {}

class MockSidebar:
    def __getattr__(self, name):
        def mock_func(*args, **kwargs):
            pass
        return mock_func

# Replace streamlit import temporarily
import sys
mock_st = MockStreamlit()
mock_st.sidebar = MockSidebar()
sys.modules['streamlit'] = mock_st

# Import the necessary functions
from ultra_fast_rag import ultra_fast_answer_question
from rag_engine import answer_question as standard_answer_question, get_cached_local_model

# Import the enhanced functions and prompt building from cluster_qa_app
def build_simple_enhanced_prompt(question, context, response_format_enum, model_type="general"):
    """Build a simple, clean prompt that models can actually follow."""
    # Get format-specific instructions
    format_instructions = ""
    if "bullet_points" in str(response_format_enum).lower():
        format_instructions = """Please provide your analysis in this bullet point format:

• **Main Finding:** [Your main answer to the question]
• **Key Evidence:** [Specific facts from the documents]  
• **Notable Patterns:** [Common themes you found]
• **Practical Implications:** [What this means for policy]"""
    elif "summary" in str(response_format_enum).lower():
        format_instructions = """Format as an executive summary with these sections:

**Primary Answer:** [Direct 1-2 sentence response to the question]

**Key Points:**
- [Key finding 1 with evidence from documents]
- [Key finding 2 with evidence from documents] 
- [Key finding 3 with evidence from documents]

**Bottom Line:** [Final recommendation or conclusion]"""
    elif "comparative" in str(response_format_enum).lower():
        format_instructions = """Format as comparative analysis with these sections:

**Overview:** [Brief statement of what is being compared]

**Similarities:**
- [Common approach 1 with evidence]
- [Common approach 2 with evidence]

**Key Differences:**
- [Difference 1 with specific examples]
- [Difference 2 with specific examples]

**Analysis:** [Why these differences exist and their significance]

**Conclusion:** [Overall assessment and recommendations]"""
    elif "technical" in str(response_format_enum).lower():
        format_instructions = """Format as technical analysis with these sections:

**Technologies/Measures:** [List specific technologies and measures mentioned]

**Quantitative Data:**
- [Target 1: specific numbers, percentages, or timelines]
- [Target 2: specific numbers, percentages, or timelines]
- [Target 3: specific numbers, percentages, or timelines]

**Implementation Details:** [How these measures are implemented]

**Effectiveness Assessment:** [Analysis of effectiveness and potential impact]"""
    else:  # Detailed
        format_instructions = """Format as detailed analysis with these sections:

**Executive Summary:** [2-3 sentence overview of key findings]

**Main Analysis:**
[Detailed explanation with supporting evidence from the documents]

**Supporting Evidence:**
- [Evidence point 1 from documents]
- [Evidence point 2 from documents]
- [Evidence point 3 from documents]

**Implications:** [What this means for policy and implementation]

**Recommendations:** [Actionable next steps or suggestions]"""

    if "deepseek" in model_type.lower():
        prompt = f"""You are a climate policy expert. Based on these UNFCCC documents, answer the question using the specified format.

{format_instructions}

Documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the documents:"""
    else:
        prompt = f"""Based on these UNFCCC documents, please answer the question using the specified format.

{format_instructions}

Documents:
{context}

Question: {question}"""

    return prompt


def enhanced_ultra_fast_answer_question(question, cluster_id, model, model_name, response_format_enum, max_tokens=8000, top_k=5):
    """Enhanced version of ultra_fast_answer_question that accepts format instructions."""
    try:
        # Get the raw answer first
        answer, passages, token_count = ultra_fast_answer_question(
            question, cluster_id, model, max_tokens, top_k
        )
        
        # If we got passages, enhance the answer with formatting
        if passages:
            context = "\n\n".join([f"Document {i+1}: {passage}" for i, passage in enumerate(passages)])
            enhanced_prompt = build_simple_enhanced_prompt(question, context, response_format_enum, model_name)
            
            # Generate enhanced answer
            result = model(
                enhanced_prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=1.1,
                top_p=0.9
            )
            
            enhanced_answer = result[0]["generated_text"].strip() if isinstance(result, list) and "generated_text" in result[0] else str(result)
            return enhanced_answer, passages, token_count
        else:
            return answer, passages, token_count
            
    except Exception as e:
        return f"Error in enhanced processing: {str(e)}", [], 0


def enhanced_answer_question(question, cluster_id, model_name, response_format_enum, top_k=5, max_tokens=4000):
    """Enhanced version of answer_question that accepts format instructions."""
    try:
        # Get the raw answer first
        answer, passages, token_count = standard_answer_question(
            question, cluster_id, model_name, max_tokens, top_k
        )
        
        # If we got passages and model is available, enhance the answer
        if passages:
            try:
                model, _ = get_cached_local_model(preferred_model_name=model_name)
                if model:
                    context = "\n\n".join([f"Document {i+1}: {passage}" for i, passage in enumerate(passages)])
                    enhanced_prompt = build_simple_enhanced_prompt(question, context, response_format_enum, model_name)
                    
                    result = model(
                        enhanced_prompt,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.1,
                        repetition_penalty=1.1,
                        top_p=0.9
                    )
                    
                    enhanced_answer = result[0]["generated_text"].strip() if isinstance(result, list) and "generated_text" in result[0] else str(result)
                    return enhanced_answer, passages, token_count
            except Exception as e:
                print(f"Warning: Could not enhance answer: {e}")
        
        return answer, passages, token_count
        
    except Exception as e:
        return f"Error in processing: {str(e)}", [], 0


def process_question_with_rag(question, cluster_id, model_name, pipeline, response_format, max_tokens, top_k, model=None):
    """Process a single question using the RAG system."""
    start_time = time.time()
    
    try:
        if pipeline == "ultra_fast":
            if model is None:
                model, _ = get_cached_local_model(preferred_model_name=model_name)
                if model is None:
                    raise Exception(f"Could not load model: {model_name}")
            
            answer, passages, token_count = enhanced_ultra_fast_answer_question(
                question, cluster_id, model, model_name, response_format, max_tokens, top_k
            )
        else:  # standard pipeline
            answer, passages, token_count = enhanced_answer_question(
                question, cluster_id, model_name, response_format, top_k, max_tokens
            )
        
        processing_time = time.time() - start_time
        return answer, passages, token_count, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error processing question: {str(e)}"
        return error_msg, [], 0, processing_time


def show_system_info():
    """Display system information similar to Streamlit app."""
    import psutil
    print("🔧 System Information:")
    print(f"   • CPU cores: {psutil.cpu_count()}")
    print(f"   • Memory: {psutil.virtual_memory().total // (1024**3)}GB total, {psutil.virtual_memory().available // (1024**3)}GB available")
    print(f"   • Python version: {sys.version.split()[0]}")

def show_openai_info():
    """Display OpenAI information if available."""
    try:
        import openai
        print(f"🔧 OpenAI Information:")
        print(f"   • OpenAI version: {openai.__version__}")
    except ImportError:
        print("🔧 OpenAI not installed")

def main():
    parser = argparse.ArgumentParser(
        description="Headless Streamlit cluster QA processor - Run RAG queries without the web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python streamlit_headless_processor.py questions.txt output.txt
  
  # Use specific cluster and format
  python streamlit_headless_processor.py questions.txt output.txt --cluster-id 5 --format bullet_points
  
  # Use faster model with country-level clustering
  python streamlit_headless_processor.py questions.txt output.txt --model deepseek-r1-distill-qwen-7b --clustering-mode country
  
  # Enable debug mode with passage information
  python streamlit_headless_processor.py questions.txt output.txt --debug --show-passage-debug
        """
    )
    parser.add_argument("questions_file", help="File containing questions (one per line)")
    parser.add_argument("output_file", help="Output file for results")
    
    # Clustering Mode
    parser.add_argument("--clustering-mode", choices=["document", "country"], default="document", 
                       help="Clustering type: document-level or country-level (default: document)")
    
    # Cluster Selection
    parser.add_argument("--cluster-id", default="all", help="Cluster ID to process (default: all)")
    
    # RAG Pipeline Selection
    parser.add_argument("--pipeline", choices=["ultra_fast", "standard"], default="ultra_fast", 
                       help="Pipeline type: ultra_fast (5-20x faster) or standard (default: ultra_fast)")
    
    # Model Selection - matching Streamlit options exactly
    parser.add_argument("--model", 
                       choices=[
                           "deepseek-r1-distill-qwen-14b",
                           "deepseek-r1-distill-qwen-7b", 
                           "deepseek-r1-distill-llama-8b",
                           "deepseek-llm-7b-chat",
                           "tinyllama-1b",
                           "distilgpt2",
                           "flan-t5-small",
                           "gpt-4o"
                       ],
                       default="deepseek-r1-distill-qwen-14b", 
                       help="AI model for answer generation (default: deepseek-r1-distill-qwen-14b)")
    
    # Response Format Selection - matching Streamlit options exactly
    parser.add_argument("--format", 
                       choices=["detailed", "summary", "bullet_points", "comparative", "technical"], 
                       default="detailed", 
                       help="Response format structure (default: detailed)")
    
    # Configuration Parameters
    parser.add_argument("--top-k", type=int, default=5, choices=range(3, 16), 
                       help="Number of top documents to retrieve (3-15, default: 5)")
    parser.add_argument("--max-tokens", type=int, default=4000, choices=range(2000, 8001, 500),
                       help="Maximum context tokens (2000-8000, default: 4000)")
    
    # Dataset Options
    parser.add_argument("--include-problematic", action="store_true", 
                       help="Include problematic documents in analysis (default: exclude)")
    
    # Debugging Options
    parser.add_argument("--show-system-info", action="store_true", 
                       help="Show technical system information")
    parser.add_argument("--show-openai-info", action="store_true", 
                       help="Show OpenAI version info")
    parser.add_argument("--show-passage-debug", action="store_true", 
                       help="Show passage debug information")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode with detailed logging")
    
    args = parser.parse_args()
    
    # Map model names to actual model identifiers (matching Streamlit app)
    model_mapping = {
        "deepseek-r1-distill-qwen-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
        "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-llm-7b-chat": "deepseek-ai/DeepSeek-LLM-7B-Chat",
        "tinyllama-1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "distilgpt2": "distilgpt2",
        "flan-t5-small": "google/flan-t5-small",
        "gpt-4o": "gpt-4o"
    }
    
    actual_model_name = model_mapping[args.model]
    
    questions_file = Path(args.questions_file)
    output_file = Path(args.output_file)
    
    # Validate input file
    if not questions_file.exists():
        print(f"❌ Questions file not found: {questions_file}")
        sys.exit(1)
    
    # Load questions
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"❌ Error reading questions file: {e}")
        sys.exit(1)
    
    if not questions:
        print("❌ No questions found in the input file")
        sys.exit(1)
    
    print(f"📝 Loaded {len(questions)} questions from {questions_file}")
    print(f"🔧 Configuration:")
    print(f"   • Clustering mode: {args.clustering_mode}-level")
    print(f"   • Cluster ID: {args.cluster_id}")
    print(f"   • Model: {args.model} ({actual_model_name})")
    print(f"   • Pipeline: {args.pipeline}")
    print(f"   • Format: {args.format}")
    print(f"   • Max tokens: {args.max_tokens}")
    print(f"   • Top-k: {args.top_k}")
    print(f"   • Include problematic docs: {args.include_problematic}")
    print(f"   • Debug mode: {args.debug}")
    
    # Show optional information
    if args.show_system_info:
        print()
        show_system_info()
    
    if args.show_openai_info:
        print()
        show_openai_info()
    
    # Load model once for ultra_fast pipeline
    model = None
    if args.pipeline == "ultra_fast":
        print(f"\n🤖 Loading model: {actual_model_name}...")
        try:
            model, loaded_model_name = get_cached_local_model(preferred_model_name=actual_model_name)
            if model is None:
                print(f"❌ Model not available: {actual_model_name}")
                sys.exit(1)
            print(f"✅ Model loaded: {loaded_model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    # Process questions
    total_start_time = time.time()
    total_processing_time = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"Streamlit Headless RAG Processing Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration:\n")
            f.write(f"• Clustering mode: {args.clustering_mode}-level\n")
            f.write(f"• Cluster ID: {args.cluster_id}\n")
            f.write(f"• Model: {args.model} ({actual_model_name})\n")
            f.write(f"• Pipeline: {args.pipeline}\n")
            f.write(f"• Response Format: {args.format}\n")
            f.write(f"• Max Tokens: {args.max_tokens}\n")
            f.write(f"• Top-K: {args.top_k}\n")
            f.write(f"• Include problematic docs: {args.include_problematic}\n")
            f.write(f"• Debug mode: {args.debug}\n")
            f.write(f"• Total Questions: {len(questions)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, question in enumerate(questions, 1):
                print(f"\n[{i}/{len(questions)}] Processing: {question[:100]}{'...' if len(question) > 100 else ''}")
                
                # Process the question
                answer, passages, token_count, processing_time = process_question_with_rag(
                    question, args.cluster_id, actual_model_name, args.pipeline, 
                    args.format, args.max_tokens, args.top_k, model
                )
                total_processing_time += processing_time
                
                print(f"✅ Completed in {processing_time:.2f}s (tokens: {token_count}, passages: {len(passages)})")
                
                # Write to output file
                f.write(f"QUESTION {i}/{len(questions)}:\n")
                f.write(f"{question}\n\n")
                f.write(f"ANSWER:\n")
                f.write(f"{answer}\n\n")
                f.write(f"METADATA:\n")
                f.write(f"• Processing time: {processing_time:.2f}s\n")
                f.write(f"• Token count: {token_count}\n")
                f.write(f"• Retrieved passages: {len(passages)}\n")
                if passages:
                    f.write(f"• Source documents:\n")
                    for j, passage in enumerate(passages[:3], 1):  # Show first 3 passages
                        preview = passage[:200] + "..." if len(passage) > 200 else passage
                        f.write(f"  {j}. {preview}\n")
                f.write("-" * 60 + "\n\n")
                
                # Flush to ensure progress is saved
                f.flush()
        
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 Batch processing completed!")
        print(f"📊 Summary:")
        print(f"   • Total questions processed: {len(questions)}")
        print(f"   • Total processing time: {total_processing_time:.2f}s")
        print(f"   • Average time per question: {total_processing_time/len(questions):.2f}s")
        print(f"   • Total elapsed time: {total_time:.2f}s")
        print(f"   • Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()