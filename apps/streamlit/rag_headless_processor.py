#!/usr/bin/env python3
"""
Headless RAG processor that replicates Streamlit app functionality.
Uses the core RAG engine without Streamlit dependencies.

Usage:
    python rag_headless_processor.py questions.txt output.txt [options]
    
Options:
    --model MODEL_NAME          Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
    --format FORMAT            Response format: detailed, bullet_points, summary, comparative, technical (default: detailed)
    --max-tokens TOKENS        Maximum tokens for response (default: 4000)
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Suppress warnings before importing heavy modules
import os
import warnings
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import core functionality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_deepseek_model(model_name):
    """Load DeepSeek model using the same logic as the Streamlit app."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        def generate_response(prompt, max_new_tokens=500, temperature=0.1, top_p=0.9, do_sample=True, repetition_penalty=1.1):
            """Generate response using the loaded model."""
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
            
            with torch.no_grad():
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            new_tokens = output[0][inputs.input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response.strip()
        
        return generate_response
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def build_formatted_prompt(question, response_format):
    """Build a prompt with specific formatting instructions."""
    format_instructions = ""
    
    if "bullet_points" in response_format.lower():
        format_instructions = """Please provide your analysis in this bullet point format:

• **Main Finding:** [Your main answer to the question]
• **Key Evidence:** [Specific facts and details]  
• **Notable Patterns:** [Common themes you found]
• **Practical Implications:** [What this means for policy]"""
    
    elif "summary" in response_format.lower():
        format_instructions = """Format as an executive summary with these sections:

**Primary Answer:** [Direct 1-2 sentence response to the question]

**Key Points:**
- [Key finding 1 with supporting details]
- [Key finding 2 with supporting details] 
- [Key finding 3 with supporting details]

**Bottom Line:** [Final recommendation or conclusion]"""
    
    elif "comparative" in response_format.lower():
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
    
    elif "technical" in response_format.lower():
        format_instructions = """Format as technical analysis with these sections:

**Technologies/Measures:** [List specific technologies and measures mentioned]

**Quantitative Data:**
- [Target 1: specific numbers, percentages, or timelines]
- [Target 2: specific numbers, percentages, or timelines]
- [Target 3: specific numbers, percentages, or timelines]

**Implementation Details:** [How these measures are implemented]

**Effectiveness Assessment:** [Analysis of effectiveness and potential impact]"""
    
    else:  # detailed
        format_instructions = """Format as detailed analysis with these sections:

**Executive Summary:** [2-3 sentence overview of key findings]

**Main Analysis:**
[Detailed explanation with supporting evidence]

**Supporting Evidence:**
- [Evidence point 1]
- [Evidence point 2]
- [Evidence point 3]

**Implications:** [What this means for policy and implementation]

**Recommendations:** [Actionable next steps or suggestions]"""

    prompt = f"""You are a climate policy expert. Please answer the following question about climate policy and UNFCCC processes using the specified format.

{format_instructions}

Question: {question}

Please provide a comprehensive answer based on your knowledge of climate policy, UNFCCC processes, and international climate negotiations:"""

    return prompt


def process_question_with_deepseek(model_func, question, response_format, max_tokens):
    """Process a single question with DeepSeek using formatted prompts."""
    start_time = time.time()
    
    try:
        # Build the formatted prompt
        prompt = build_formatted_prompt(question, response_format)
        
        # Generate response
        answer = model_func(
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
        
        processing_time = time.time() - start_time
        return answer, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Error processing question: {str(e)}"
        return error_msg, processing_time


def main():
    parser = argparse.ArgumentParser(description="Headless RAG processor for climate policy questions")
    parser.add_argument("questions_file", help="File containing questions (one per line)")
    parser.add_argument("output_file", help="Output file for results")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", help="Model name")
    parser.add_argument("--format", choices=["detailed", "bullet_points", "summary", "comparative", "technical"], 
                       default="detailed", help="Response format")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Maximum tokens for response")
    
    args = parser.parse_args()
    
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
    print(f"   • Model: {args.model}")
    print(f"   • Format: {args.format}")
    print(f"   • Max tokens: {args.max_tokens}")
    
    # Load model
    print(f"\n🤖 Loading model: {args.model}...")
    model_func = get_deepseek_model(args.model)
    if model_func is None:
        print(f"❌ Failed to load model: {args.model}")
        sys.exit(1)
    print(f"✅ Model loaded successfully")
    
    # Process questions
    total_start_time = time.time()
    total_processing_time = 0
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write header
            f.write(f"RAG Headless Processing Results\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Response Format: {args.format}\n")
            f.write(f"Max Tokens: {args.max_tokens}\n")
            f.write(f"Total Questions: {len(questions)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, question in enumerate(questions, 1):
                print(f"\n[{i}/{len(questions)}] Processing: {question[:100]}{'...' if len(question) > 100 else ''}")
                
                # Process the question
                answer, processing_time = process_question_with_deepseek(
                    model_func, question, args.format, args.max_tokens
                )
                total_processing_time += processing_time
                
                print(f"✅ Completed in {processing_time:.2f}s")
                
                # Write to output file
                f.write(f"QUESTION {i}/{len(questions)}:\n")
                f.write(f"{question}\n\n")
                f.write(f"ANSWER:\n")
                f.write(f"{answer}\n\n")
                f.write(f"METADATA:\n")
                f.write(f"• Processing time: {processing_time:.2f}s\n")
                f.write(f"• Response format: {args.format}\n")
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