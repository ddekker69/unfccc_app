"""
UNFCCC Climate QA System - Advanced Prompt Templates
===================================================

This module contains prompt templates with detailed formatting instructions to improve
LLM response quality, consistency, and structure.

Features:
- Structured response formats
- Context-aware templates
- Model-specific optimizations
- Clear output guidelines
"""

import logging
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    """Supported response formats for different use cases."""
    DETAILED = "detailed"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"
    COMPARATIVE = "comparative"
    TECHNICAL = "technical"

class PromptTemplate:
    """Advanced prompt template with structured formatting instructions."""
    
    @staticmethod
    def get_system_instructions(format_type: ResponseFormat = ResponseFormat.DETAILED) -> str:
        """
        Get comprehensive system instructions for the AI assistant.
        
        Args:
            format_type: The desired response format type
            
        Returns:
            Formatted system instructions string
        """
        base_instructions = """You are a specialized AI assistant for climate policy analysis. Your role is to provide accurate, well-structured, and actionable insights from UNFCCC climate documents.

CORE RESPONSIBILITIES:
• Analyze climate policy documents with precision and depth
• Synthesize information from multiple sources coherently
• Provide evidence-based answers with proper context
• Maintain objectivity and scientific accuracy
• Focus on practical implications and policy relevance

OUTPUT QUALITY STANDARDS:
• Use clear, professional language appropriate for policy makers
• Structure responses logically with proper flow between ideas
• Support statements with specific evidence from the documents
• Distinguish between facts, interpretations, and recommendations
• Acknowledge limitations and uncertainties when relevant"""

        format_instructions = {
            ResponseFormat.DETAILED: """
RESPONSE FORMAT - DETAILED ANALYSIS:
1. **Executive Summary** (2-3 sentences)
   - Key finding or direct answer to the question
   - Most critical insight or recommendation

2. **Main Analysis** (structured paragraphs)
   - Detailed explanation with supporting evidence
   - Reference specific documents, countries, or policies
   - Include relevant context and background

3. **Supporting Evidence**
   - Quote or reference specific document excerpts
   - Mention countries, dates, or specific commitments
   - Highlight patterns or trends across documents

4. **Implications & Context**
   - Practical significance of the findings
   - How this relates to broader climate goals
   - Potential challenges or opportunities

5. **Key Takeaways** (bullet points)
   - 3-5 actionable insights
   - Clear, concise statements
   - Prioritized by importance""",

            ResponseFormat.SUMMARY: """
RESPONSE FORMAT - EXECUTIVE SUMMARY:
1. **Primary Answer** (1-2 sentences)
   - Direct response to the question
   - Most important finding

2. **Key Points** (3-4 bullet points)
   - Essential information from the documents
   - Specific examples or evidence
   - Notable patterns or exceptions

3. **Bottom Line** (1 sentence)
   - Practical implication or recommendation""",

            ResponseFormat.BULLET_POINTS: """
RESPONSE FORMAT - STRUCTURED BULLETS:
• **Main Finding:** Clear statement of the primary answer
• **Key Evidence:** 
  - Supporting fact 1 (with source/country reference)
  - Supporting fact 2 (with source/country reference)
  - Supporting fact 3 (with source/country reference)
• **Notable Patterns:** Common themes or trends identified
• **Exceptions/Variations:** Important differences or unique cases
• **Practical Implications:** What this means for policy or implementation""",

            ResponseFormat.COMPARATIVE: """
RESPONSE FORMAT - COMPARATIVE ANALYSIS:
1. **Overview:** Brief statement of what's being compared

2. **Similarities:**
   - Common approaches or shared commitments
   - Consistent themes across documents/countries

3. **Key Differences:**
   - Varying approaches or priorities
   - Different implementation strategies
   - Contrasting commitments or timelines

4. **Analysis:**
   - Why these differences exist
   - Implications of the variations
   - Best practices or effective approaches

5. **Synthesis:** Overall assessment and recommendations""",

            ResponseFormat.TECHNICAL: """
RESPONSE FORMAT - TECHNICAL ANALYSIS:
1. **Technical Summary:** Precise answer with specific metrics/data

2. **Methodology/Approach:**
   - How countries/policies address the technical issue
   - Specific technologies, measures, or frameworks mentioned

3. **Quantitative Data:**
   - Numbers, targets, percentages, or timelines
   - Comparative metrics across countries/documents

4. **Implementation Details:**
   - Specific mechanisms, policies, or procedures
   - Technical requirements or standards

5. **Assessment:**
   - Effectiveness or feasibility analysis
   - Technical challenges or barriers identified"""
        }

        return base_instructions + "\n" + format_instructions.get(format_type, format_instructions[ResponseFormat.DETAILED])

    @staticmethod
    def build_enhanced_prompt(
        question: str, 
        context: str, 
        format_type: ResponseFormat = ResponseFormat.DETAILED,
        model_type: str = "general",
        additional_instructions: Optional[str] = None
    ) -> str:
        """
        Build a comprehensive prompt with formatting instructions.
        
        Args:
            question: The user's question
            context: Retrieved document context
            format_type: Desired response format
            model_type: Type of model ("deepseek", "openai", "general")
            additional_instructions: Optional extra instructions
            
        Returns:
            Complete formatted prompt
        """
        system_instructions = PromptTemplate.get_system_instructions(format_type)
        
        # Model-specific formatting
        if "deepseek" in model_type.lower() or "r1" in model_type.lower():
            prompt = f"""<|im_start|>system
{system_instructions}

IMPORTANT FORMATTING GUIDELINES:
• Start responses immediately with substantive content
• Use markdown formatting for structure (##, **, •)
• Keep explanations clear and focused
• Cite specific countries or document details when relevant
• End with actionable insights
<|im_end|>
<|im_start|>user
Based on the following excerpts from UNFCCC climate policy documents, please answer the question with the specified format and quality standards.

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

{additional_instructions if additional_instructions else ""}

Please provide a comprehensive, well-structured response following the format guidelines above.
<|im_end|>
<|im_start|>assistant
"""
        elif "openai" in model_type.lower() or "gpt" in model_type.lower():
            prompt = f"""Based on the following excerpts from UNFCCC climate policy documents, please answer the question with high quality and structure.

{system_instructions}

FORMATTING GUIDELINES:
• Use clear section headers with ##
• Support points with specific evidence
• Include country names and specific details
• Structure for easy reading and action

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

{additional_instructions if additional_instructions else ""}

Please provide a comprehensive, well-structured response following the format guidelines above."""
        else:
            # General format for other models
            prompt = f"""Human: You are a climate policy expert. Based on the following relevant excerpts from UNFCCC documents, please answer the question with high quality and structure.

GUIDELINES:
{system_instructions}

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

{additional_instructions if additional_instructions else ""}

Please provide a comprehensive analysis that synthesizes information from multiple sources and follows the specified format."""
        
        return prompt